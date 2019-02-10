import random
import magent
from magent.builtin.rule_model import RandomActor

if __name__ == "__main__":
    gw = magent.gridworld
    cfg = gw.Config()

    map_size = 25
    cfg.set({"map_width": map_size, "map_height": map_size})

    up = cfg.register_agent_type(
        "up",
        {"restrict_movement": True,
         "movement_direction": 0
        })
    right = cfg.register_agent_type(
        "right",
        {"restrict_movement": True,
         "movement_direction": 3
        })
    
    upgroup = cfg.add_group(up)
    rightgroup = cfg.add_group(right)
    
    # add reward rule
    a = gw.AgentSymbol(upgroup, index='any')
    b = gw.AgentSymbol(rightgroup, index='any')
    e1 = gw.Event(a, 'in', ((9,0), (15,2)))
    e2 = gw.Event(b, 'in', ((map_size-3,9), (map_size-1,15)))
    e3 = gw.Event(a, 'collide', b)
    cfg.add_reward_rule(e1, receiver=a, value=1, die=True)
    cfg.add_reward_rule(e2, receiver=b, value=1, die=True)
    cfg.add_reward_rule(e3, receiver=[a,b], value=[-1,-1])

    env = magent.GridWorld(cfg)

    upgroup, rightgroup = env.get_handles()
    model1 = RandomActor(env, upgroup, "up")
    model2 = RandomActor(env, rightgroup, "right")
    # print(env.get_action_space(rightgroup))

    env.set_render_dir("build/render")

    env.reset()
    upstart = [(x, map_size-2) for x in range(10,15)]
    rightstart = [(1, y) for y in range(10,15)]
    spawnrate = 0.1
    env.add_agents(upgroup, method="custom", pos=upstart)
    env.add_agents(rightgroup, method="custom", pos=rightstart)

    done = False
    step_ct = 0
    while not done:

        # agent comes in random rate
        upspawn = [random.random() < spawnrate for _ in range(5)]
        rightspawn = [random.random() < spawnrate for _ in range(5)]
        env.add_agents(upgroup, method="custom", pos=[p for p,f in zip(upstart,upspawn) if f is True])
        env.add_agents(rightgroup, method="custom", pos=[p for p,f in zip(rightstart,rightspawn) if f is True])

        obs_1 = env.get_observation(upgroup)
        ids_1 = env.get_agent_id(upgroup)
        acts_1 = model1.infer_action(obs_1, ids_1)
        env.set_action(upgroup, acts_1)

        obs_2 = env.get_observation(rightgroup)
        ids_2 = env.get_agent_id(rightgroup)
        acts_2 = model2.infer_action(obs_2, ids_2)
        env.set_action(rightgroup, acts_2)

        # simulate one step
        done = env.step()

        # render
        env.render()

        # get reward
        reward = [sum(env.get_reward(upgroup)), sum(env.get_reward(rightgroup))]

        # clear dead agents
        env.clear_dead()

        # print info
        # if step_ct % 10 == 0:
        #     print("step %d" % step_ct)
        step_ct += 1
        if step_ct > 250:
            break