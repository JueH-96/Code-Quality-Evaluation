import sys

def main():
    import sys
    import sys
    from collections import defaultdict

    input = sys.stdin.read().split()
    N = int(input[0])
    A_C = input[1:]
    color_min = {}
    for i in range(N):
        A = int(A_C[2*i])
        C = int(A_C[2*i +1])
        if C in color_min:
            if A < color_min[C]:
                color_min[C] = A
        else:
            color_min[C] = A
    result = max(color_min.values())
    print(result)

if __name__ == "__main__":
    main()