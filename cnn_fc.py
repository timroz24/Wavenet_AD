import os

import numpy as np
import tensorflow as tf
import pandas as pd
import itertools
import glob

from data_frame_fc import DataGenerator
from tf_base_model_fc import TFBaseModel
from tf_utils import (
    time_distributed_dense_layer, temporal_convolution_layer, find_missing_timesteps_idx_per_equipment,
    sequence_mean, sequence_smape,sequence_smape_new, shape, get_target_data_cols, find_good_idx_equipment
)


class DataReader(object):    
    def __init__(self, data_cols, target_sensors, gas_type_ids,data_dir):
        print("******* Gettting File Names/Paths **********")
        self.equipment_id_dict = {}
        self.equipment_missing_idx = {}
        self.equipment_df_resampled = {}
        self.gas_type_ids = gas_type_ids
        data_cols, no_of_target_sensors = get_target_data_cols(data_cols,target_sensors)
        self.data_cols = data_cols
        print("No of Columns : ", len(self.data_cols))
        self.no_of_target_sensors = no_of_target_sensors
        self.target_sensors = target_sensors
        self.get_all_file_paths(data_dir)
        print("******* Loading Files **********")
        self.load_all_equipment_files()
        self.DataGenerator = DataGenerator(self.equipment_id_dict)
        self.DataGenerator.train_test_split(self.equipment_df_resampled, train_size=0.88)
        print('train size', len(self.DataGenerator.train_idx))
        print('val size', len(self.DataGenerator.test_idx))
        
    def read_parquet_files(self, file_path):
        return pd.read_parquet(file_path,columns=self.data_cols)
    
    def get_all_file_paths(self, data_dir): 
        min_days_to_keep = 10
        all_equipment = glob.glob(data_dir +'_*')
        all_equipment_files = {}
        all_date_files = []
        for date_dir in all_equipment:
            date_file_paths = []
            for root, directories, files in os.walk(date_dir, topdown=False):
                for filename in files: 
                    filepath = os.path.join(root, filename)
                    date_file_paths.append(filepath)
            data_file_paths = sorted(date_file_paths)
            if len(data_file_paths)>=min_days_to_keep:
                all_equipment_files[date_dir[-11:]] = data_file_paths    
        self.equipment_files_dict = all_equipment_files
        
        
    def load_all_equipment_files(self):        
        equip_id_count = 0
        for key, value in self.equipment_files_dict.items():            
            if int(key) in self.gas_type_ids:
                print('Equip_No : {}  Key : {} '.format(equip_id_count, key))                
                self.load_df(key,value)
                self.equipment_id_dict[equip_id_count] = key
                equip_id_count +=1
#                 if kk==70:
#                     break

                
                
        
        
    def load_df(self, equipment_id,equipment_file_paths):
        equipment_df = []
        for i,file_path in enumerate(equipment_file_paths):
            equipment_df.append(self.read_parquet_files(file_path))
        equipment_df = pd.concat(equipment_df,axis=0)
        equipment_df.index = equipment_df['date_time']
        equipment_df.index = equipment_df.index.map(lambda x: x.replace(second=0))
        equipment_df = equipment_df.drop('date_time',axis=1)
        self.equipment_df_resampled[equipment_id], self.equipment_missing_idx[equipment_id] = find_missing_timesteps_idx_per_equipment(equipment_id, equipment_df, resample='1Min')
                

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            batch_type='Train',
            shuffle=True,
            num_epochs=10000,
            is_test=False
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            batch_type='Val',
            shuffle=True,
            num_epochs=10000,
            is_test=False
        )

#     def test_batch_generator(self, batch_size):
#         return self.batch_generator(
#             batch_size=batch_size,
#             df=self.test_df,
#             shuffle=True,
#             num_epochs=1,
#             is_test=True
#         )

    def batch_generator(self, batch_size,batch_type, shuffle=True, num_epochs=10000, is_test=False):
        batch_gen = self.DataGenerator.batch_generator(
            batch_size=batch_size,
            batch_type=batch_type,
            shuffle=shuffle,
            num_epochs=num_epochs,
            allow_smaller_final_batch=is_test
        )
        data_col = 'test_data' if is_test else 'data'
        batch_dict = {}
        for batch in batch_gen:
            num_encode_steps = 192 #### one day in minutes
            num_decode_steps = 32
#             max_encode_length = self.full_seq_len - num_decode_steps if not is_test else full_seq_len
            max_encode_length = num_encode_steps - num_decode_steps
            

            x_encode = np.zeros([len(batch), max_encode_length])
            y_decode = np.zeros([len(batch), num_decode_steps])
            encode_len = np.zeros([len(batch)])
            decode_len = np.zeros([len(batch)])
            ### 1 column for removed data_time & 1 column for x_encode target sensor
            encode_features = np.zeros([len(batch),max_encode_length,len(self.data_cols)-2]) 
            decode_features = np.zeros([len(batch),num_decode_steps,len(self.data_cols)-2])
            
            seq_len_list = [] 
            target_col_list = []
            good_idx_list = []
            for i, equip_id in enumerate(batch):
                seq_len_list.append(np.random.randint(num_decode_steps*2, num_encode_steps))
                target_col_list.append(np.random.randint(0,self.no_of_target_sensors))
#                 target_col_list.append(0)
                good_idx_list.append(find_good_idx_equipment(self.equipment_missing_idx[equip_id], seq_len_list[i]))
                
            for j in range(500):            
                for i, equip_id in enumerate(batch):

                    #### Generate random seq. length
                    seq_len = seq_len_list[i]
                    #### Generate random target col            
                    target_col = target_col_list[i]
                    good_idx = good_idx_list[i]
                    start_idx = good_idx[np.random.randint(len(good_idx))]
                    seq = self.equipment_df_resampled[equip_id]
                    
#                     print(" Target_Col ",target_col, " seq_len ",seq_len, " start_idx ", start_idx)

                      ### look at how to optimize this            
                    x_encode_len = max_encode_length if is_test else seq_len - num_decode_steps #### change this later for handling is_test == True
                    x_encode[i, 0:x_encode_len] = seq[start_idx:start_idx + x_encode_len,target_col] 
                    if not is_test:
                        y_decode[i, :] = seq[start_idx + x_encode_len: start_idx + x_encode_len + num_decode_steps,target_col]

                    seq = np.delete(seq, target_col, axis=1)

                    encode_len[i] = x_encode_len
                    decode_len[i] = num_decode_steps

                    encode_features[i,0:x_encode_len,:] = seq[start_idx:start_idx + x_encode_len,:] 

                    if not is_test:                    
                        decode_features[i,0:num_decode_steps,:] =  seq[start_idx + x_encode_len: start_idx + x_encode_len + num_decode_steps,:]            

                batch_dict['x_encode'] = x_encode
                batch_dict['encode_len'] = encode_len
                batch_dict['y_decode'] = y_decode
                batch_dict['decode_len'] = decode_len
                batch_dict['encode_features'] = encode_features
                batch_dict['decode_features'] = decode_features           
                yield batch_dict
            
            
class cnn(TFBaseModel):

    def __init__(
        self,
        residual_channels=32,
        skip_channels=32,
        dilations=[2**i for i in range(8)]*3,
        filter_widths=[2 for i in range(8)]*3,
        num_decode_steps=32,
        **kwargs
    ):
        
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilations = dilations
        self.filter_widths = filter_widths
        self.num_decode_steps = num_decode_steps
        super(cnn, self).__init__(**kwargs)

    def transform(self, x, is_features=False):
        if is_features:
            return tf.log(x + 1) - tf.expand_dims(self.log_encode_features_mean, 1)
        else:
            return tf.log(x + 1) - tf.expand_dims(self.log_x_encode_mean, 1)    
    

    def inverse_transform(self, x):
        return tf.exp(x + tf.expand_dims(self.log_x_encode_mean, 1)) - 1

    def get_input_sequences(self):
        self.x_encode = tf.placeholder(tf.float32, [None, None])
        self.encode_len = tf.placeholder(tf.int32, [None])
        self.y_decode = tf.placeholder(tf.float32, [None, self.num_decode_steps])
        self.decode_len = tf.placeholder(tf.int32, [None])       

        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)

        self.log_x_encode_mean = sequence_mean(tf.log(self.x_encode + 1), self.encode_len)
#         print(" self.log_x_encode_mean " ,self.log_x_encode_mean)
        self.log_x_encode = self.transform(self.x_encode)
        self.x = tf.expand_dims(self.log_x_encode, 2)               
        
        self.encode_features = tf.placeholder(tf.float32, [None, None, len(self.reader.data_cols) - 2 ])       
        feature_encode_len = tf.tile(tf.expand_dims(self.encode_len,axis=1),(1,self.encode_features.shape[2]))        
        self.log_encode_features_mean = sequence_mean(tf.log(self.encode_features + 1), feature_encode_len)
        self.encode_features = self.transform(self.encode_features, is_features=True)
        
        ##### Add code to transform features ######
#         self.encode_features = #### Add code here
        
        ##### Delete this chunk once self.encode_features are transformed
#         self.encode_features = tf.concat([
#             tf.expand_dims(tf.cast(tf.equal(self.x_encode, 0.0), tf.float32), 2),
#             tf.tile(tf.reshape(self.log_x_encode_mean, (-1, 1, 1)), (1, tf.shape(self.x_encode)[1], 1)),
#             tf.tile(tf.expand_dims(tf.one_hot(self.project, 9), 1), (1, tf.shape(self.x_encode)[1], 1)),
#             tf.tile(tf.expand_dims(tf.one_hot(self.access, 3), 1), (1, tf.shape(self.x_encode)[1], 1)),
#             tf.tile(tf.expand_dims(tf.one_hot(self.agent, 2), 1), (1, tf.shape(self.x_encode)[1], 1)),
#         ], axis=2)
        
#         self.encode_features = tf.placeholder(tf.float32, [None, None])    
#         self.encode_features = tf.expand_dims(self.encode_features, 2)
    
#         self.encode_features = tf.concat([self.x,self.x,], axis=2)
        
        ###### Walk through this code to see if needed anymore?
#         decode_idx = tf.tile(tf.expand_dims(tf.range(self.num_decode_steps), 0), (tf.shape(self.y_decode)[0], 1))
#         self.decode_features = tf.concat([
#             tf.one_hot(decode_idx, self.num_decode_steps),
#             tf.tile(tf.reshape(self.log_x_encode_mean, (-1, 1, 1)), (1, self.num_decode_steps, 1)),
#             tf.tile(tf.expand_dims(tf.one_hot(self.project, 9), 1), (1, self.num_decode_steps, 1)),
#             tf.tile(tf.expand_dims(tf.one_hot(self.access, 3), 1), (1, self.num_decode_steps, 1)),
#             tf.tile(tf.expand_dims(tf.one_hot(self.agent, 2), 1), (1, self.num_decode_steps, 1)),
#         ], axis=2)

        self.decode_features = tf.placeholder(tf.float32, [None, self.num_decode_steps, len(self.reader.data_cols) - 2 ])
        self.decode_features = self.transform(self.decode_features, is_features=True)
        
#         decode_idx = tf.tile(tf.expand_dims(tf.range(self.num_decode_steps), 0), (tf.shape(self.y_decode)[0], 1))
#         self.decode_features = tf.concat([
#             tf.one_hot(decode_idx, self.num_decode_steps),
#             self.x,
#         ], axis=2)

        return self.x

    def encode(self, x, features):
        x = tf.concat([x, features], axis=2)

        inputs = time_distributed_dense_layer(
            inputs=x,
            output_units=self.residual_channels,
            activation=tf.nn.tanh,
            scope='x-proj-encode'
        )

        skip_outputs = []
        conv_inputs = [inputs]
        for i, (dilation, filter_width) in enumerate(zip(self.dilations, self.filter_widths)):
            dilated_conv = temporal_convolution_layer(
                inputs=inputs,
                output_units=2*self.residual_channels,
                convolution_width=filter_width,
                causal=True,
                dilation_rate=[dilation],
                scope='dilated-conv-encode-{}'.format(i)
            )
            conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
            dilated_conv = tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)

            outputs = time_distributed_dense_layer(
                inputs=dilated_conv,
                output_units=self.skip_channels + self.residual_channels,
                scope='dilated-conv-proj-encode-{}'.format(i)
            )
            skips, residuals = tf.split(outputs, [self.skip_channels, self.residual_channels], axis=2)

            inputs += residuals
            conv_inputs.append(inputs)
            skip_outputs.append(skips)

        skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))
        h = time_distributed_dense_layer(skip_outputs, 128, scope='dense-encode-1', activation=tf.nn.relu)
        y_hat = time_distributed_dense_layer(h, 1, scope='dense-encode-2')

        return y_hat, conv_inputs[:-1]

    def initialize_decode_params(self, x, features):
        x = tf.concat([x, features], axis=2)

        inputs = time_distributed_dense_layer(
            inputs=x,
            output_units=self.residual_channels,
            activation=tf.nn.tanh,
            scope='x-proj-decode'
        )

        skip_outputs = []
        conv_inputs = [inputs]
        for i, (dilation, filter_width) in enumerate(zip(self.dilations, self.filter_widths)):
            dilated_conv = temporal_convolution_layer(
                inputs=inputs,
                output_units=2*self.residual_channels,
                convolution_width=filter_width,
                causal=True,
                dilation_rate=[dilation],
                scope='dilated-conv-decode-{}'.format(i)
            )
            conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
            dilated_conv = tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)

            outputs = time_distributed_dense_layer(
                inputs=dilated_conv,
                output_units=self.skip_channels + self.residual_channels,
                scope='dilated-conv-proj-decode-{}'.format(i)
            )
            skips, residuals = tf.split(outputs, [self.skip_channels, self.residual_channels], axis=2)

            inputs += residuals
            conv_inputs.append(inputs)
            skip_outputs.append(skips)

        skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=2))
        h = time_distributed_dense_layer(skip_outputs, 128, scope='dense-decode-1', activation=tf.nn.relu)
        y_hat = time_distributed_dense_layer(h, 1, scope='dense-decode-2')
        return y_hat

    def decode(self, x, conv_inputs, features):
        batch_size = tf.shape(x)[0]

        # initialize state tensor arrays
        state_queues = []
        for i, (conv_input, dilation) in enumerate(zip(conv_inputs, self.dilations)):
            batch_idx = tf.range(batch_size)
            batch_idx = tf.tile(tf.expand_dims(batch_idx, 1), (1, dilation))
            batch_idx = tf.reshape(batch_idx, [-1])

            queue_begin_time = self.encode_len - dilation - 1
            temporal_idx = tf.expand_dims(queue_begin_time, 1) + tf.expand_dims(tf.range(dilation), 0)
            temporal_idx = tf.reshape(temporal_idx, [-1])

            idx = tf.stack([batch_idx, temporal_idx], axis=1)
            slices = tf.reshape(tf.gather_nd(conv_input, idx), (batch_size, dilation, shape(conv_input, 2)))

            layer_ta = tf.TensorArray(dtype=tf.float32, size=dilation + self.num_decode_steps)
            layer_ta = layer_ta.unstack(tf.transpose(slices, (1, 0, 2)))
            state_queues.append(layer_ta)

        # initialize feature tensor array
        features_ta = tf.TensorArray(dtype=tf.float32, size=self.num_decode_steps)
        features_ta = features_ta.unstack(tf.transpose(features, (1, 0, 2)))

        # initialize output tensor array
        emit_ta = tf.TensorArray(size=self.num_decode_steps, dtype=tf.float32)

        # initialize other loop vars
        elements_finished = 0 >= self.decode_len
        time = tf.constant(0, dtype=tf.int32)

        # get initial x input
        current_idx = tf.stack([tf.range(tf.shape(self.encode_len)[0]), self.encode_len - 1], axis=1)
        initial_input = tf.gather_nd(x, current_idx)

        def loop_fn(time, current_input, queues):
            current_features = features_ta.read(time)
            current_input = tf.concat([current_input, current_features], axis=1)

            with tf.variable_scope('x-proj-decode', reuse=True):
                w_x_proj = tf.get_variable('weights')
                b_x_proj = tf.get_variable('biases')
                x_proj = tf.nn.tanh(tf.matmul(current_input, w_x_proj) + b_x_proj)

            skip_outputs, updated_queues = [], []
            for i, (conv_input, queue, dilation) in enumerate(zip(conv_inputs, queues, self.dilations)):

                state = queue.read(time)
                with tf.variable_scope('dilated-conv-decode-{}'.format(i), reuse=True):
                    w_conv = tf.get_variable('weights'.format(i))
                    b_conv = tf.get_variable('biases'.format(i))
                    dilated_conv = tf.matmul(state, w_conv[0, :, :]) + tf.matmul(x_proj, w_conv[1, :, :]) + b_conv
                conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=1)
                dilated_conv = tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)

                with tf.variable_scope('dilated-conv-proj-decode-{}'.format(i), reuse=True):
                    w_proj = tf.get_variable('weights'.format(i))
                    b_proj = tf.get_variable('biases'.format(i))
                    concat_outputs = tf.matmul(dilated_conv, w_proj) + b_proj
                skips, residuals = tf.split(concat_outputs, [self.skip_channels, self.residual_channels], axis=1)

                x_proj += residuals
                skip_outputs.append(skips)
                updated_queues.append(queue.write(time + dilation, x_proj))

            skip_outputs = tf.nn.relu(tf.concat(skip_outputs, axis=1))
            with tf.variable_scope('dense-decode-1', reuse=True):
                w_h = tf.get_variable('weights')
                b_h = tf.get_variable('biases')
                h = tf.nn.relu(tf.matmul(skip_outputs, w_h) + b_h)

            with tf.variable_scope('dense-decode-2', reuse=True):
                w_y = tf.get_variable('weights')
                b_y = tf.get_variable('biases')
                y_hat = tf.matmul(h, w_y) + b_y

            elements_finished = (time >= self.decode_len)
            finished = tf.reduce_all(elements_finished)

            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, 1], dtype=tf.float32),
                lambda: y_hat
            )
            next_elements_finished = (time >= self.decode_len - 1)

            return (next_elements_finished, next_input, updated_queues)

        def condition(unused_time, elements_finished, *_):
            return tf.logical_not(tf.reduce_all(elements_finished))

        def body(time, elements_finished, emit_ta, *state_queues):
            (next_finished, emit_output, state_queues) = loop_fn(time, initial_input, state_queues)

            emit = tf.where(elements_finished, tf.zeros_like(emit_output), emit_output)
            emit_ta = emit_ta.write(time, emit)

            elements_finished = tf.logical_or(elements_finished, next_finished)
            return [time + 1, elements_finished, emit_ta] + list(state_queues)

        returned = tf.while_loop(
            cond=condition,
            body=body,
            loop_vars=[time, elements_finished, emit_ta] + state_queues
        )

        outputs_ta = returned[2]
        y_hat = tf.transpose(outputs_ta.stack(), (1, 0, 2))
        return y_hat

    def calculate_loss(self):
        x = self.get_input_sequences()

        y_hat_encode, conv_inputs = self.encode(x, features=self.encode_features)
        self.initialize_decode_params(x, features=self.decode_features)
        y_hat_decode = self.decode(y_hat_encode, conv_inputs, features=self.decode_features)
        y_hat_decode = self.inverse_transform(tf.squeeze(y_hat_decode, 2))
        y_hat_decode = tf.nn.relu(y_hat_decode)

        self.labels = self.y_decode
        self.preds = y_hat_decode
        self.loss = sequence_smape(self.labels, self.preds, self.decode_len)

        self.prediction_tensors = {
            'priors': self.x_encode,
            'labels': self.labels,
            'preds': self.preds,
            #### Equip_ids_list
        }

        return self.loss