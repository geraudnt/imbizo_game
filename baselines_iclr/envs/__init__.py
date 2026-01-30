from gym.envs.registration import register

# ----------------------------------------- Half-Cheetah

register(
    id='Half-Cheetah-v0',
    entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnv',
)
register(
    id='Half-Cheetah-RM1-v0',
    entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM1',
    max_episode_steps=1000,
)
register(
    id='Half-Cheetah-RM2-v0',
    entry_point='envs.mujoco_rm.half_cheetah_environment:MyHalfCheetahEnvRM2',
    max_episode_steps=1000,
)


# ----------------------------------------- WATER
register(
    id='PickUpObj-v0',
    entry_point='envs.gym_minigrid.pickupobjs:PickUpObjEnv',
)

register(
    id='PickUpObj-RM1-v0',
    entry_point='envs.gym_minigrid.pickupobjs:PickUpObjEnvRM1',
    max_episode_steps=1000
)

# ----------------------------------------- WATER
register(
    id='Water-v0',
    entry_point='envs.water.water_environment:WaterEnv',
)

for i in range(11):
    w_id = 'Water-M%d-v0'%i
    w_en = 'envs.water.water_environment:WaterRMEnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=600
    )

for i in range(11):
    w_id = 'Water-single-M%d-v0'%i
    w_en = 'envs.water.water_environment:WaterRM10EnvM%d'%i
    register(
        id=w_id,
        entry_point=w_en,
        max_episode_steps=600
    )

# ----------------------------------------- OFFICE
register(
    id='Office-v0',
    entry_point='envs.grids.grid_environment:OfficeEnv',
)

register(
    id='Office-multiple-v0',
    entry_point='envs.grids.grid_environment:OfficeRMEnv',
    max_episode_steps=1000
)

register(
    id='Office-single-v0',
    entry_point='envs.grids.grid_environment:OfficeRM3Env',
    max_episode_steps=1000
)

register(
    id='Office-new-v0',
    entry_point='envs.office_world_modified.GridWorld:GridWorldEnv',
)

register(
    id='Office-new-multiple-v0',
    entry_point='envs.office_world_modified.GridWorld:GridWorldRMEnv',
    max_episode_steps=1000
)

register(
    id='Office-new-rm1-v0',
    entry_point='envs.office_world_modified.GridWorld:GridWorldRM1Env',
    max_episode_steps=1000
)

register(
    id='Office-new-rm2-v0',
    entry_point='envs.office_world_modified.GridWorld:GridWorldRM2Env',
    max_episode_steps=1000
)

register(
    id='Office-new-rm3-v0',
    entry_point='envs.office_world_modified.GridWorld:GridWorldRM3Env',
    max_episode_steps=1000
)

register(
    id='Office-new-rm4-v0',
    entry_point='envs.office_world_modified.GridWorld:GridWorldRM4Env',
    max_episode_steps=1000
)

register(
    id='Office-new-rm5-v0',
    entry_point='envs.office_world_modified.GridWorld:GridWorldRM5Env',
    max_episode_steps=1000
)


register(
    id='Office-new-ue-v0',
    entry_point='envs.office_world_modified.GridWorld_unsatisfiable_env:GridWorldEnv',
)

register(
    id='Office-new-ue-multiple-v0',
    entry_point='envs.office_world_modified.GridWorld_unsatisfiable_env:GridWorldRMEnv',
    max_episode_steps=1000
)

register(
    id='Office-new-ue-rm1-v0',
    entry_point='envs.office_world_modified.GridWorld_unsatisfiable_env:GridWorldRM1Env',
    max_episode_steps=1000
)

register(
    id='Office-new-ue-rm2-v0',
    entry_point='envs.office_world_modified.GridWorld_unsatisfiable_env:GridWorldRM2Env',
    max_episode_steps=1000
)

register(
    id='Office-new-ue-rm3-v0',
    entry_point='envs.office_world_modified.GridWorld_unsatisfiable_env:GridWorldRM3Env',
    max_episode_steps=1000
)

register(
    id='Office-new-ue-rm4-v0',
    entry_point='envs.office_world_modified.GridWorld_unsatisfiable_env:GridWorldRM4Env',
    max_episode_steps=1000
)

register(
    id='Office-new-ue-rm5-v0',
    entry_point='envs.office_world_modified.GridWorld_unsatisfiable_env:GridWorldRM5Env',
    max_episode_steps=1000
)


register(
    id='Office-new-up-v0',
    entry_point='envs.office_world_modified.GridWorld_unsatisfiable_prop:GridWorldEnv',
)

register(
    id='Office-new-up-multiple-v0',
    entry_point='envs.office_world_modified.GridWorld_unsatisfiable_prop:GridWorldRMEnv',
    max_episode_steps=1000
)

register(
    id='Office-new-up-rm1-v0',
    entry_point='envs.office_world_modified.GridWorld_unsatisfiable_prop:GridWorldRM1Env',
    max_episode_steps=1000
)

register(
    id='Office-new-up-rm2-v0',
    entry_point='envs.office_world_modified.GridWorld_unsatisfiable_prop:GridWorldRM2Env',
    max_episode_steps=1000
)

register(
    id='Office-new-up-rm3-v0',
    entry_point='envs.office_world_modified.GridWorld_unsatisfiable_prop:GridWorldRM3Env',
    max_episode_steps=1000
)

register(
    id='Office-new-up-rm4-v0',
    entry_point='envs.office_world_modified.GridWorld_unsatisfiable_prop:GridWorldRM4Env',
    max_episode_steps=1000
)

register(
    id='Office-new-up-rm5-v0',
    entry_point='envs.office_world_modified.GridWorld_unsatisfiable_prop:GridWorldRM5Env',
    max_episode_steps=1000
)