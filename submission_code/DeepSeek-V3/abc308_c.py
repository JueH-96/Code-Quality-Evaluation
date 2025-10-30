# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = []
    B = []
    index = 1
    for _ in range(N):
        A.append(int(data[index]))
        B.append(int(data[index+1]))
        index += 2
    # Calculate success rates as A_i / (A_i + B_i)
    # To avoid floating point inaccuracies, compare A_i * (A_j + B_j) vs A_j * (A_i + B_i)
    # So, we can sort based on the cross product
    # We need to sort the indices based on the success rate in descending order
    # So, for two indices i and j, if A_i * (A_j + B_j) > A_j * (A_i + B_i), then i comes before j
    # To implement this, we can create a list of tuples (A_i, A_i + B_i, i) and sort based on the cross product
    people = []
    for i in range(N):
        total = A[i] + B[i]
        people.append((A[i], total, i+1))  # storing i+1 as the person number
    # Sort based on the cross product A_i * total_j > A_j * total_i
    # To sort in descending order, we can use the negative of the cross product
    # So, for two tuples (a1, t1, idx1) and (a2, t2, idx2), we compare a1 * t2 vs a2 * t1
    # To sort in descending order, we can use the key as (a1 * t2 - a2 * t1)
    # But to avoid floating points, we can use the cross product directly
    # So, for sorting, we can use the key as (a1 * t2 - a2 * t1)
    # But to sort in descending order, we can use the negative of the cross product
    # So, we can define a custom comparator
    # In Python, the sorted function allows a key function, but for custom comparators, we need to use the cmp_to_key function
    from functools import cmp_to_key
    def compare(x, y):
        a1, t1, idx1 = x
        a2, t2, idx2 = y
        cross1 = a1 * t2
        cross2 = a2 * t1
        if cross1 > cross2:
            return -1
        elif cross1 < cross2:
            return 1
        else:
            if idx1 < idx2:
                return -1
            else:
                return 1
    people_sorted = sorted(people, key=cmp_to_key(compare))
    # Extract the indices in order
    result = [person[2] for person in people_sorted]
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()