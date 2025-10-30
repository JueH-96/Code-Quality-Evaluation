from collections import Counter

def min_operations(a, k):
    n = len(a)
    operations = 0
    
    # Count the frequency of each number in the array
    freq = Counter(a)
    
    # Iterate through the numbers and perform operations to make them divisible by k
    for num, count in freq.items():
        if num % k != 0:
            # Find the minimum number of operations needed to make the number divisible by k
            operations += count * (k - num % k)
    
    return operations

t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    print(min_operations(a, k))