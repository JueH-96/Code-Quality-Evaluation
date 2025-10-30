# YOUR CODE HERE
def solve():
    N, M = map(int, input().split())
    S = input().strip()
    max_k = 0
    A = 0
    B = 0
    for c in S:
        if c == '0':
            needed = B + max(A - M, 0)
            if needed > max_k:
                max_k = needed
            A = 0
            B = 0
        else:
            if c == '1':
                A +=1
            elif c == '2':
                B +=1
    # Process the last period
    needed = B + max(A - M, 0)
    if needed > max_k:
        max_k = needed
    print(max_k if max_k >0 else 0)