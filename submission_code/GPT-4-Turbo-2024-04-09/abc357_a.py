def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    H = list(map(int, data[2:]))
    
    count = 0
    remaining_disinfectant = M
    
    for hands_needed in H:
        if remaining_disinfectant >= hands_needed:
            remaining_disinfectant -= hands_needed
            count += 1
        else:
            break
    
    print(count)

if __name__ == "__main__":
    main()