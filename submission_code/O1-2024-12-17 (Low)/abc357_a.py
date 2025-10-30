def main():
    import sys
    data = sys.stdin.read().strip().split()
    
    N = int(data[0])
    M = int(data[1])
    H = list(map(int, data[2:]))

    count = 0
    remaining_disinfectant = M
    
    for hands in H:
        if remaining_disinfectant >= hands:
            count += 1
            remaining_disinfectant -= hands
        else:
            # The alien will use up what's left but won't disinfect all hands
            remaining_disinfectant = 0
            # No need to continue once disinfectant is used up, but the problem 
            # still wants us to consider all aliens, so we do not break.
    
    print(count)

# Do not remove this call
main()