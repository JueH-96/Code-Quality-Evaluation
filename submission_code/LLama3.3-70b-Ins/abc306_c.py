import sys

def calculate_f(A, N):
    """
    Calculate the index of the middle occurrence of each number in A.

    Args:
    A (list): The input sequence.
    N (int): The number of unique elements in A.

    Returns:
    dict: A dictionary where the keys are the unique elements in A and the values are their middle indices.
    """
    f = {}
    for i in range(1, N + 1):
        indices = [j for j, x in enumerate(A) if x == i]
        f[i] = indices[1] + 1  # Add 1 because indices are 0-based
    return f

def sort_by_f(f):
    """
    Sort the keys of f by their values.

    Args:
    f (dict): A dictionary where the keys are the unique elements and the values are their middle indices.

    Returns:
    list: A list of keys sorted by their values.
    """
    return sorted(f, key=f.get)

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    f = calculate_f(A, N)
    sorted_keys = sort_by_f(f)
    print(' '.join(map(str, sorted_keys)))

if __name__ == "__main__":
    main()