def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    P = list(map(int, input_data[1:N+1]))
    
    first_person = P[0]
    max_other = max(P[1:]) if N > 1 else 0
    
    # calculate the minimum x needed
    if first_person > max_other:
        print(0)
    else:
        # x needed such that first_person + x > max_other
        x = max_other - first_person + 1
        print(x)

if __name__ == '__main__':
    main()