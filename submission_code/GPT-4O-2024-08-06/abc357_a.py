# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    H = list(map(int, data[2:]))
    
    remaining_disinfectant = M
    count = 0
    
    for hands in H:
        if remaining_disinfectant >= hands:
            count += 1
            remaining_disinfectant -= hands
        else:
            break
    
    print(count)

if __name__ == "__main__":
    main()