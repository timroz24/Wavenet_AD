import os
from cnn_fc import DataReader, cnn

lpg_equipment_instance_ids = [
    20170402060,20170701834,20170600627,20170300094,20170401951,20170800651,20170501814,20170401932,
    20170402063,20170402165,20170401491,20170402053,20170400629,20170601904,20170600610,20170600844,
    20170901905,20170501105,20170402050,20171001234,20170701831,20170800268,20171200688,20170801262,
    20170601422,20170401937,20170600861,20170501096,20170801627,20171001898,20170401429,20170401466,
    20170901022,20170702101,20170801267,20171100863,20171001887,20171100554,20170600630,20171200566,
    20170501101,20171101404,20170801266,20171200687,20170800629,20171100551,20170901906,20171001243,
    20170701172,20170800895,20171100862,20171200561,20171200689,20170801626,20171101396,20170402067,
    20170501823,20170602143,20171101398,20170900649,20171001885,20171001903,20171000890,20171100860,
    20170901631,20171000905,20170800901,20170602122,20170602603,20170901634,20171001221,20170401485,
    20170800632,20170501115,20170600629,20170701833,20171001223,20170401418,20170701198,20170701312,
    20170701181,20171101606,20170602375,20170300104,20170401939,20170401430,20170300098,20170300112,
    20170401471,20170400622,20170400601,20170300090,20170402048,20170602617,20170501466,20170401930,
    20170501118,20170602388,20170701195,20170501470,20170602609,20170601426,20170401477,20170300114,
    20170402171,20170401406,20170601430,20170800652,20170600852,20170501451,20170602361,20170901035,
    20170800638,20170601441,20170701180,20171200696,20171101407,20171100214,20170800883,20170901016,
    20170901902,20170901007,20171200703,20170600144,20171200275,20170401929,20170402058,20170602139,
    20170600848,20171200576,20170401468,20171100857,20170901026,20171000900,20171100864,20170602373,
    20171001238,20170300109,20170800899,20170401408,20170401486,20170701825,20170901645,20170601433,
    20170801276,20170400620,20171200679,20171100871,20171200587,20170701826,20170900643,20170501472,
    20170601911,20170600604,20171100239
]

lpg_equipment_instance_ids = [
    20170402060,20170701834,20170600627
]



data_dir = '/home/arimo/data/tim/atomic/'
file_prefix = 'equipment'
date_time = ['date_time']

# Water
## Water navy blue

r_tank = ['r_tank_th'] # float32
r_circulation_pump = [
    'r_circulation_p_rotation_number_target_value',  # int32
    'r_circulation_p_operation_amount', # float32
    'r_circulation_p_rotational_speed' # int32
    
]
r_circulation_pump_stack = [
    'fc_input_th_target_value', # float32
    'fc_inlet_temperature' # float32
]
stack_ = [
    'fc_output_th_target_value_fc_output_temperature_target', # float32
    'fc_outlet_temperature' # float32
]

## Water lightblue

cd_tank = [
    'cd_tank_temperature' # float32
]
reform_water_pump = [
#     'detection_of_reforming_water_p_rotation_detection', # bool
    'reformed_water_flow_rate', # float32
    'reformed_water_flow_rate_target_value', # float32
    'reforming_water_pump_operation_amount' # float32 
]
# reform_water_valve = [
#     'reforming_water_valve' # bool
# ]
# air_vent_valve = [
#     'air_vent_valve' # bool
# ]

## Water magenta 

hot_water_storage_in = [
    'sys_inlet_temperature' # float32
]

hot_water_storage_out = [
    'sys_outlet_temperature' # float32
]

c_circulation_pump = [
    'c_circulation_p_rotational_speed', # int32
    'c_circulation_p_rotation_number_target_value', # int32
    'c_circulation_p_operation_amount' # float32
]


# Air
## Air green_1

air_blower = [
    'air_blower_operation_amount' # float32
]


# air_inlet_valve = [
#     'air_inlet_valve_1' # bool
# ]
#"why no sensors for this?" check with ACCDC team
# air_outlet_valve = []

p_flow_meter = [
    'p_flow_rate', # float32
    'p_flow_rate_target_value' # float32
]

# prox_air_valve = [
#     'p_air_valve' # bool
# ]

## Air Green_2

n_flow_meter = [
    'n_flow_rate_target_value', # float32
    'combustion_air_flow_rate' # float32
]

bb_fan_completion_a = [
    'bb_fan_speed', # int32
    'br_thermocouple_target_value_for_combustion_fan_control', # float32
    'bb_fan_operation_amount' # float32
]

## Air gray has no sensors

# Gas

gas = [
#     'power_supply_control_of_gas_pressure_gauge', # bool
    'gas_base_pressure', # float32
#     'gas_source_pressure_valve_a', # bool
#     'gas_source_pressure_valve_b'  # bool 
]

gas_source_meter = [
    'cumulative_flow_rate_of_raw_material_gas' # int32
]

booster_pump = [
    'booster_pump_operation_amount', # float32
    'raw_material_flow_rate_fc_unit_consumption_gas', # float32
    'g_flow_rate_target_value', # float32
    'br_thermocouple_target_value_for_material_control' # float32
]

# igniter = [
#     'igniter_ignition_transformer', # bool
#     'igniter_fb_igniter_fb_signal'  # bool
    
# ]

## Fuel Processor
fuel_processor = [
    'bp_inlet_temperature', # float32
    'bh_thermocouple_temperature', # float32
    'br_thermocouple_temperature', # float32
#     'bh_bp_heater', # bool
#     'bh_bp_heater_fb' # bool
]

## Stack
stack = [
    'stack_current' # float32
]
## Power generation

power_generation = [
    'fc_power_generation_amount',
    'auxiliary_machine_power_fc_unit_power_consumption',
    'backward_flow_prevention_heater_power',
    'power_target_low_limit_set_value_power_up_protection_control',
    'lower_limit_of_power_generation_command_measures_against_19_f',
    'load_power',
    'pv_generated_electric_power_photovoltaic_power_generation',
    'purchased_electric_power',
    'pcs_input_current_direct_current',
    'generated_electric_power_command_value_a_v'
]



# selected_cols =  date_time + r_tank + r_circulation_pump + r_circulation_pump_stack + stack_ + cd_tank + reform_water_pump +\
# hot_water_storage_in + hot_water_storage_out + c_circulation_pump +\
# air_blower + p_flow_meter + n_flow_meter + bb_fan_completion_a +\
# gas + gas_source_meter + booster_pump + fuel_processor + stack + power_generation

schema =  date_time + booster_pump + gas + gas_source_meter + fuel_processor + stack

target_sensors = schema[1:]

if __name__ == '__main__':
    base_dir = './'
    dr = DataReader(schema, target_sensors, lpg_equipment_instance_ids,data_dir=os.path.join(data_dir, file_prefix))
    nn = cnn(
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs'),
        checkpoint_dir=os.path.join(base_dir, 'new_checkpoints'),
        prediction_dir=os.path.join(base_dir, 'predictions'),
        optimizer='adam',
        learning_rate=0.001,
        batch_size=8,
        num_training_steps=20000,
        early_stopping_steps=5000,
        warm_start_init_step=0,
        regularization_constant=0.0001,
        keep_prob=1.0,
        enable_parameter_averaging=False,
        num_restarts=2,
        min_steps_to_checkpoint=500,
        log_interval=10,
        num_validation_batches=1,
        grad_clip=20,
        residual_channels=32,
        skip_channels=32,
        dilations=[2**i for i in range(8)]*3,
        filter_widths=[2 for i in range(8)]*3,
        num_decode_steps=32,
    )
    nn.fit()
