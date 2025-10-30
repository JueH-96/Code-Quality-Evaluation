def main():
    import sys
    H = int(sys.stdin.readline().strip())
    
    # We'll find the smallest i such that (2^i - 1) > H.
    # Start with i = 1 (because day 0 morning height is 0).
    # We keep increasing i until 2^i - 1 > H.
    
    i = 1
    while True:
        if (1 << i) - 1 > H:  # (1 << i) is 2^i
            print(i)
            break
        i += 1

# DO NOT forget to call main()
main()