from collections import defaultdict

def find_max_1122_sequence_length(arr):
    """
    Finds the maximum length of a contiguous subarray of arr that is a 1122 sequence.
    """
    num_to_indices = defaultdict(list)
    for i, num in enumerate(arr):
        num_to_indices[num].append(i)
    
    best_length = 0
    seen = set()
    for i, num in enumerate(arr):
        if num not in seen:
            if len(num_to_indices[num]) == 2:
                if i < num_to_indices[num][1]:
                    best_length = max(best_length, num_to_indices[num][1] - i)
                    seen.add(num)
                elif num_to_indices[num][1] < i and i < num_to_indices[num][0]:
                    best_length = max(best_length, num_to_indices[num][0] - i)
                    seen.add(num)
        else:
            seen.remove(num)
    
    i, j, new_best_length = 0, 0, 0
    seen = set()
    while j < len(arr):
        if arr[j] not in seen:
            seen.add(arr[j])
            while len(num_to_indices[arr[j]]) == 2 and i < num_to_indices[arr[j]][1]:
                seen.remove(arr[i])
                i += 1
            new_best_length = max(new_best_length, j - i + 1)
        j += 1
    
    return max(best_length, new_best_length)

# Read input
N = int(input())
A = list(map(int, input().split()))

# Solve and print the answer
print(find_max_1122_sequence_length(A))