# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    data = N_and_rest[1:]
    products = []
    for i in range(N):
        T = int(data[2*i])
        D = int(data[2*i +1])
        products.append( (T + D, T) )
    products.sort()
    count =0
    last_print_time = -1 << 60
    for end, start in products:
        p = max(start, last_print_time +1)
        if p <= end:
            count +=1
            last_print_time = p
    print(count)

if __name__ == "__main__":
    main()