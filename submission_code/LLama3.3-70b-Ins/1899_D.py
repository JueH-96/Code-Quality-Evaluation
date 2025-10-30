import sys

def count_pairs(n, a):
    """
    Count the number of pairs (i, j) (i < j) such that b_i^{b_j} = b_j^{b_i}.
    
    Args:
    n (int): The length of the array.
    a (list): The array of integers.
    
    Returns:
    int: The number of pairs that satisfy the given condition.
    """
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Check if b_i^{b_j} = b_j^{b_i}
            if a[i] * 2**a[j] == a[j] * 2**a[i]:
                count += 1
    return count

def main():
    t = int(input())
    for _ in range(t):
        n = int(input())
        a = list(map(int, input().split()))
        print(count_pairs(n, a))

if __name__ == "__main__":
    main()