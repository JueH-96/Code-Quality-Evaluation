import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    player_data = []
    # Step 1 & 2: Read input and pre-compute scores and unsolved problems for each player
    for i in range(N): # Player i (0-indexed) is Player i+1 in problem statement
        s_line = sys.stdin.readline().strip()
        
        # Calculate current score for player i
        # Bonus score for player (i+1) is (i+1) points
        current_score = (i + 1) 
        
        unsolved_problem_scores_for_player_i = []
        
        for j in range(M): # Problem j (0-indexed)
            if s_line[j] == 'o':
                current_score += A[j]
            else:
                unsolved_problem_scores_for_player_i.append(A[j])
        
        # Sort this player's unsolved problems by score in descending order
        unsolved_problem_scores_for_player_i.sort(reverse=True) 
        
        player_data.append({
            'id': i, 
            'initial_score': current_score,
            'unsolved_scores': unsolved_problem_scores_for_player_i
        })

    # Step 3: For each player P, calculate the minimum number of additional problems to solve
    output_lines = []
    for i in range(N): # Consider player P (represented by player_data[i])
        player_P_info = player_data[i]
        p_initial_score = player_P_info['initial_score']
        p_unsolved_scores_list = player_P_info['unsolved_scores']
        
        # Find the maximum score among all *other* players
        max_other_current_score = 0
        for j in range(N):
            if i == j: # Don't compare player P with themselves
                continue
            max_other_current_score = max(max_other_current_score, player_data[j]['initial_score'])
            
        # If player P's current score is already strictly greater than all others
        if p_initial_score > max_other_current_score:
            output_lines.append("0")
        else:
            # Player P needs to score more.
            # Target score: 1 more than max_other_current_score
            target_score_for_P = max_other_current_score + 1
            additional_score_needed = target_score_for_P - p_initial_score
            
            # additional_score_needed will be >= 1 because p_initial_score <= max_other_current_score

            num_problems_to_solve = 0
            accumulated_score_from_new_problems = 0
            
            for problem_score_value in p_unsolved_scores_list:
                accumulated_score_from_new_problems += problem_score_value
                num_problems_to_solve += 1
                if accumulated_score_from_new_problems >= additional_score_needed:
                    break 
            # The problem guarantees a solution exists by solving all unsolved problems.
            # So, the loop will always break before running out of problems.
            output_lines.append(str(num_problems_to_solve))
            
    sys.stdout.write("
".join(output_lines) + "
")

if __name__ == '__main__':
    main()