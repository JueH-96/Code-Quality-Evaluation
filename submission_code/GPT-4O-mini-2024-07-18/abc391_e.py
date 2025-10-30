def majority_operation(B):
    n = len(B) // 3
    C = []
    for i in range(n):
        group = B[3*i:3*i+3]
        count_0 = group.count('0')
        count_1 = group.count('1')
        if count_1 > count_0:
            C.append('1')
        else:
            C.append('0')
    return ''.join(C)

def final_majority(A):
    while len(A) > 1:
        A = majority_operation(A)
    return A

def count_changes_to_flip(A, target):
    n = len(A)
    changes_needed = 0
    for i in range(n):
        if A[i] == target:
            changes_needed += 1
    return changes_needed

def min_changes_to_flip_final_value(N, A):
    # Get the final majority value
    final_value = final_majority(A)
    # Determine the target value we want to flip to
    target_value = '0' if final_value == '1' else '1'
    
    # Count how many changes are needed to flip the final value
    changes_needed = count_changes_to_flip(A, target_value)
    
    return changes_needed

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = data[1]

# Calculate the result
result = min_changes_to_flip_final_value(N, A)

# Print the result
print(result)