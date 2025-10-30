from collections import Counter

def min_operations_to_divisible_by_k(n, k, a):
    """
    Finds the minimum number of operations needed to make the product of all the numbers in the array divisible by k.
    """
    operations = 0
    if k == 2:
        for num in a:
            if num % 2 != 0:
                operations += 1
    elif k == 3:
        for num in a:
            if num % 3 != 0:
                operations += 1
        operations = min(operations, n - operations + 1)
    elif k == 4:
        for num in a:
            if num % 4 != 0:
                operations += 1
        operations = min(operations, n - operations + 2)
    elif k == 5:
        for num in a:
            if num % 5 != 0:
                operations += 1
        operations = min(operations, n - operations + 4)
    return operations

def main():
    t = int(input())
    for _ in range(t):
        n, k = map(int, input().split())
        a = list(map(int, input().split()))
        print(min_operations_to_divisible_by_k(n, k, a))

if __name__ == "__main__":
    main()