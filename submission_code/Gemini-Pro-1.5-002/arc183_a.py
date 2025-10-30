# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())

    def count_good_sequences(current_seq, counts):
        if len(current_seq) == n * k:
            return 1
        
        total_count = 0
        for i in range(1, n + 1):
            if counts[i - 1] < k:
                new_counts = list(counts)
                new_counts[i - 1] += 1
                total_count += count_good_sequences(current_seq + [i], new_counts)
        return total_count

    def find_kth_sequence(target_index, current_seq, counts):
        if len(current_seq) == n * k:
            return current_seq
        
        current_index = 0
        for i in range(1, n + 1):
            if counts[i - 1] < k:
                new_counts = list(counts)
                new_counts[i - 1] += 1
                count = count_good_sequences(current_seq + [i], new_counts)
                
                if current_index + count >= target_index:
                    return find_kth_sequence(target_index - current_index, current_seq + [i], new_counts)
                else:
                    current_index += count
        return []

    s = count_good_sequences([], [0] * n)
    target_index = (s + 1) // 2
    result = find_kth_sequence(target_index, [], [0] * n)
    print(*result)

solve()