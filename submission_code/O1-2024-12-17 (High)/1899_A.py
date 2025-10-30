def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    idx = 1
    for _ in range(t):
        n = int(input_data[idx]); idx+=1
        # If n is divisible by 3, Vanya is forced to move away from multiples of 3 
        # and Vova can always move it back. Otherwise Vanya can win immediately.
        if n % 3 == 0:
            print("Second")
        else:
            print("First")

# Do not forget to call main()!
main()