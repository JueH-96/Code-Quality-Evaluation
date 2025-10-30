import math

def solve():
    n, k = map(int, input().split())
    initial_counts = [k] * n
    total_length = n * k
    target_rank = (math.factorial(total_length) // math.prod([math.factorial(k) for _ in range(n)]))
    target_rank = (target_rank + 1) // 2
    current_counts = list(initial_counts)
    result_sequence = []
    remaining_length = total_length
    
    while remaining_length > 0:
        for number in range(1, n + 1):
            if current_counts[number-1] > 0:
                current_counts[number-1] -= 1
                sequences_starting_with_number = 0
                current_sum_counts = sum(current_counts)
                if current_sum_counts >= 0:
                    numerator = math.factorial(current_sum_counts)
                    denominator = 1
                    for count in current_counts:
                        denominator *= math.factorial(count)
                    if denominator > 0:
                        sequences_starting_with_number = numerator // denominator
                    else:
                        sequences_starting_with_number = 0
                else:
                    sequences_starting_with_number = 0
                    
                if target_rank <= sequences_starting_with_number:
                    result_sequence.append(number)
                    remaining_length -= 1
                    break
                else:
                    target_rank -= sequences_starting_with_number
                    current_counts[number-1] += 1 
                    
    print(*(result_sequence))

if __name__ == '__main__':
    solve()