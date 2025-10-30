def determine_winner(n, results):
    total_t_wins = results.count('T')
    total_a_wins = results.count('A')
    
    if total_t_wins > total_a_wins:
        return 'T'
    elif total_a_wins > total_t_wins:
        return 'A'
    else:  # total_t_wins == total_a_wins
        t_wins = 0
        a_wins = 0
        
        for i, result in enumerate(results):
            if result == 'T':
                t_wins += 1
                if t_wins == total_t_wins:
                    t_reached_at = i
                    break
        
        for i, result in enumerate(results):
            if result == 'A':
                a_wins += 1
                if a_wins == total_a_wins:
                    a_reached_at = i
                    break
        
        if t_reached_at < a_reached_at:
            return 'T'
        else:
            return 'A'

# Read input
n = int(input())
results = input()

# Determine and output the winner
print(determine_winner(n, results))