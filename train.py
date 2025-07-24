import os
from agent import FishGymEnv, make_agent

def main():
    os.makedirs("models", exist_ok=True)

    # Create environment (set render=True to watch)
    env = FishGymEnv(render=False)

    # Create or load agent
    model_path = "models/fish_swim"
    if os.path.exists(model_path + ".zip"):
        model = make_agent(env)
        model = model.load(model_path, env=env)
        print("Loaded existing model.")
    else:
        model = make_agent(env)

    # Train
    TIMESTEPS = 1_000_000
    model.learn(total_timesteps=TIMESTEPS)

    # Save
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

if __name__ == "__main__":
    main()