def main():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    idx = 1
    
    for _ in range(t):
        n = int(data[idx]); idx += 1
        
        # If n % 3 == 0, Vanya ("First") cannot immediately make it divisible by 3
        # on his turn. Vova can always return it to a multiple of 3 on his own turn,
        # preventing Vanya from winning within 10 moves.
        # Otherwise, if n % 3 != 0, Vanya can win immediately (by subtracting 1 if n%3==1,
        # or adding 1 if n%3==2).
        
        if n % 3 == 0:
            print("Second")
        else:
            print("First")

# Do not forget to call main()
main()