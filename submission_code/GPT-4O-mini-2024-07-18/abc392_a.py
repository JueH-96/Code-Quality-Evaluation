def can_form_product(A):
    # Generate all permutations of A
    from itertools import permutations
    
    for B in permutations(A):
        if B[0] * B[1] == B[2]:
            return "Yes"
    return "No"

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    A = list(map(int, input().strip().split()))
    result = can_form_product(A)
    print(result)