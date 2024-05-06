import subprocess
import time
import pandas as pd


def run_test(agent, depth, num_ghosts, layout):
    with open('scores.csv', 'w') as file:
        file.write('')

    command = f"py pacman.py -p {agent} -a depth={depth} -l {layout} --frameTime 0 -k {num_ghosts}"
    start_time = time.time()
    process = subprocess.Popen(
        command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    end_time = time.time()

    execution_time = end_time - start_time
    with open('scores.csv', 'r') as file:
        lines = file.readlines()
    line0 = lines[0].strip()
    parts = line0.split(",")
    # Remove the \n
    parts[-1] = parts[-1].strip()
    score = parts[0]
    result = parts[1]

    return score, result, execution_time


def collect_results(iterations=20, configurations=None, path_to='pacman_results.csv'):
    results = []

    if not configurations:
        print('❌ No configurations provided')
        return

    for agent, depth, layout, num_ghosts in configurations:
        for _ in range(iterations):
            score, result, time_taken = run_test(
                agent, depth, num_ghosts, layout)
            results.append({
                'Agent': agent,
                'Depth': depth,
                'Layout': layout,
                'Ghosts': num_ghosts,
                'Score': score,
                'Result': result,
                'Execution Time': time_taken
            })
            print(f'✅ [{_+1}/{iterations}] Test completed:',
                  agent, depth, layout, num_ghosts)

    results_df = pd.DataFrame(results)
    results_df.to_csv(path_to, index=False)


# Run the test collection

configurations = [
    # Agent, Depth, Layout, Ghosts
    ('AlphaBetaAgent', 1, 'smallClassic', 1),
    ('AlphaBetaAgent', 1, 'smallClassic', 2),
    ('AlphaBetaAgent', 2, 'smallClassic', 1),
    ('AlphaBetaAgent', 2, 'smallClassic', 2),
    ('AlphaBetaAgent', 3, 'smallClassic', 1),
    ('AlphaBetaAgent', 3, 'smallClassic', 2),
    ('AlphaBetaAgent', 3, 'mediumClassic', 2),
    ('MinimaxAgent', 1, 'smallClassic', 1),
    ('MinimaxAgent', 1, 'smallClassic', 2),
    ('MinimaxAgent', 2, 'smallClassic', 1),
    ('MinimaxAgent', 2, 'smallClassic', 2),
    ('MinimaxAgent', 3, 'smallClassic', 1),
    ('MinimaxAgent', 3, 'smallClassic', 2),
    ('MinimaxAgent', 3, 'mediumClassic', 2),
]

collect_results(iterations=5, configurations=configurations,
                path_to='pacman_results.csv')
