def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    actions = [(int(data[i*2+1]), data[i*2+2]) for i in range(N)]
    
    # Initialize positions of left and right hands to None
    left_pos = None
    right_pos = None
    
    fatigue = 0
    
    for key, hand in actions:
        if hand == 'L':
            if left_pos is None:
                left_pos = key
            else:
                fatigue += abs(left_pos - key)
                left_pos = key
        elif hand == 'R':
            if right_pos is None:
                right_pos = key
            else:
                fatigue += abs(right_pos - key)
                right_pos = key
    
    print(fatigue)

if __name__ == "__main__":
    main()