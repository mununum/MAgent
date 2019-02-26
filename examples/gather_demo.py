import random
import magent
from magent.builtin.rule_model import RandomActor
import numpy as np



def init_food(env, food_handle):
    tree = np.asarray([[-1,0], [0,0], [0,-1], [0,1], [1,0]])
    third = map_size//4 # mapsize includes walls
    for i in range(1, 4):
        for j in range(1, 4):
            base = np.asarray([third*i, third*j])
            env.add_agents(food_handle, method="custom", pos=tree+base)

def neigbor_regen_food(env, food_handle, p=0.003):
    coords = env.get_pos(food_handle)

    rands = np.random.random(len(coords))
    for i, pos in enumerate(coords):
        if rands[i] > p:
            continue
        neighbor = np.asarray([[-1,0],[0,-1], [0,1], [1,0]])
        regen_pos = [pos+neighbor[np.random.randint(0,4)]]

        env.add_agents(food_handle, method="custom", 
            pos=regen_pos)

if __name__ == "__main__":
    gw = magent.gridworld
    cfg = gw.Config()

    map_size = 25
    cfg.set({"map_width": map_size, "map_height": map_size})

    agent_group = cfg.add_group(
        cfg.register_agent_type(
            name="agent",
            attr={
                'width': 1, 
                'length': 1, 
                'view_range': gw.CircleRange(4),
                'can_gather': True}))

    food_group = cfg.add_group(
        cfg.register_agent_type(
        "food",
        attr={'width': 1, 
             'length': 1,
             'can_be_gathered': True}))


    # add reward rule
    a = gw.AgentSymbol(agent_group, index='any')
    b = gw.AgentSymbol(food_group, index='any')
    e = gw.Event(a, 'collide', b)
    cfg.add_reward_rule(e, receiver=a, value=1)
    # cfg.add_reward_rule(e2, receiver=b, value=1, die=True)
    # cfg.add_reward_rule(e3, receiver=[a,b], value=[-1,-1])

    env = magent.GridWorld(cfg)
    agent_handle, food_handle = env.get_handles()

    model1 = RandomActor(env, agent_handle, "up")
    env.set_render_dir("build/render")

    env.reset()
    

    upstart = [(map_size//2 - 2, map_size//2 - 2), (map_size//2 + 2, map_size//2 - 2), 
            (map_size//2, map_size//2), (map_size//2 - 2, map_size//2 + 2),
            (map_size//2 + 2, map_size//2 + 2)]

    # spawnrate = 0.1
    env.add_agents(agent_handle, method="custom", pos=upstart)
    # env.add_agents(rightgroup, method="custom", pos=rightstart)
    init_food(env, food_handle)

    k = env.get_observation(agent_handle)

    print env.get_pos(agent_handle)
    print len(env.get_pos(food_handle))

    done = False
    step_ct = 0
    r_sum = 0
    while not done:
        obs_1 = env.get_observation(agent_handle)
        ids_1 = env.get_agent_id(agent_handle)
        acts_1 = model1.infer_action(obs_1, ids_1)
        env.set_action(agent_handle, acts_1)

        # simulate one step
        done = env.step()

        # render
        env.render()

        # get reward
        reward = sum(env.get_reward(agent_handle))
        r_sum += reward

        # clear dead agents
        env.clear_dead()
        neigbor_regen_food(env, food_handle)

        # print info
        # if step_ct % 10 == 0:
        #     print("step %d" % step_ct)
        step_ct += 1
        if step_ct > 250:
            break

    print r_sum