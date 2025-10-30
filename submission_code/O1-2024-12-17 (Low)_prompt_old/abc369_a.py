def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    A, B = map(int, data)

    # The possible values of x that allow A, B, x to be
    # rearranged into an arithmetic sequence are derived from:
    #
    # 1) (A, B, x) => 2*B = A + x => x = 2B - A
    # 2) (A, x, B) => 2*x = A + B => x = (A + B)/2 (only if A + B is even)
    # 3) (B, A, x) => 2*A = B + x => x = 2A - B
    #
    # These three expressions (with duplicates removed) give
    # us all candidates for x.
    
    answers = set()
    answers.add(2 * B - A)  # from case (A, B, x) or (x, B, A)
    answers.add(2 * A - B)  # from case (B, A, x) or (x, A, B)
    if (A + B) % 2 == 0:    # from case (A, x, B) or (B, x, A), if it is an integer
        answers.add((A + B) // 2)
    
    print(len(answers))

def main():
    solve()

if __name__ == "__main__":
    main()