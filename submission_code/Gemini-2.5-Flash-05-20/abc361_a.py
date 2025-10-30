import sys

def solve():
    # Read N, K, X from the first line
    n, k, x = map(int, sys.stdin.readline().split())

    # Read the sequence A from the second line
    a = list(map(int, sys.stdin.readline().split()))

    # Insert X immediately after the K-th element.
    # In the problem statement, K is 1-indexed.
    # To insert an element X after the K-th element (1-indexed),
    # we need to insert it at index K in a 0-indexed list.
    # For example, if K=3, we want to insert X after a[2] (the 3rd element).
    # This means X should be placed at index 3, shifting subsequent elements.
    # Python's list.insert(index, element) method inserts 'element' BEFORE 'index'.
    # So, list.insert(K, X) will correctly place X at index K,
    # effectively putting it after the original K-th element (at index K-1).
    a.insert(k, x)

    # Print the modified sequence B, with elements separated by spaces
    print(*(a))

if __name__ == '__main__':
    solve()