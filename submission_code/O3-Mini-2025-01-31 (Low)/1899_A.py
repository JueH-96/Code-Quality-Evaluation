def main():
    import sys
    data = sys.stdin.read().split()
    t = int(data[0])
    results = []
    for i in range(1, t+1):
        n = int(data[i])
        # Vanya can immediately win if the number mod 3 is 1 or 2 by moving to 0.
        # Otherwise when n mod 3 == 0, he cannot win on his move.
        if n % 3 == 0:
            results.append("Second")
        else:
            results.append("First")
    sys.stdout.write("
".join(results))
    
if __name__ == '__main__':
    main()