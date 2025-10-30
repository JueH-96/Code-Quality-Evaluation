def can_form_pyramid(arr, k):
    target_len = 2 * k - 1
    if len(arr) < target_len:
        return False
    
    # Generate target pyramid of size k
    target = []
    for i in range(1, k + 1):
        target.append(i)
    for i in range(k - 1, 0, -1):
        target.append(i)
    
    # Check all possible subarrays of length target_len
    for start in range(len(arr) - target_len + 1):
        # Check if this subarray can be transformed to target pyramid
        can_transform = True
        for i in range(target_len):
            if arr[start + i] < target[i]:
                can_transform = False
                break
        
        if can_transform:
            return True
    
    return False

n = int(input())
a = list(map(int, input().split()))

# Find maximum k by trying from largest possible down to 1
max_possible_k = (n + 1) // 2
result = 1

for k in range(max_possible_k, 0, -1):
    if can_form_pyramid(a, k):
        result = k
        break

print(result)