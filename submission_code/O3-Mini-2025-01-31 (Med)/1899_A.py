def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    t = int(data[0])
    results = []
    
    # The winning strategy is as follows:
    # Vanya wins immediately on his move if the current number modulo 3 is 1 or 2,
    # because he can add or subtract 1 to obtain 0 mod 3.
    # However, if the number is initially divisible by 3, then Vanya is forced to move
    # to either 1 or 2 (both losing positions since Vova can then move back to 0).
    # Thus Vanya wins ("First") when n mod 3 is not 0, and loses ("Second") otherwise.
    
    for i in range(1, t + 1):
        n = int(data[i])
        if n % 3 == 0:
            results.append("Second")
        else:
            results.append("First")
    
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()