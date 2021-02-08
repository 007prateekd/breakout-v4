from train import *

frames = []
def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode = "rgb_array"))

prev_lives = tf_env.pyenv.envs[0].ale.lives()
def reset_and_fire_on_life_lost(trajectory):
    global prev_lives
    lives = tf_env.pyenv.envs[0].ale.lives()
    if prev_lives != lives:
        tf_env.reset()
        tf_env.pyenv.envs[0].step(np.array(1))
        prev_lives = lives

watch_driver = DynamicStepDriver(
    tf_env,
    agent.policy,
    observers = [save_frames, reset_and_fire_on_life_lost, init.ShowProgress(1000)],
    num_steps = 1000)

final_time_step, final_policy_state = watch_driver.run()    
image_path = os.path.join("Breakout.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames[:150]]
frame_images[0].save(
    image_path, format = "GIF",
    append_images=frame_images[1:],
    save_all = True, duration = 30, loop = 0)