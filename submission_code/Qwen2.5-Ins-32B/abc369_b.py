import sys

def min_fatigue_level(N, presses):
    left_hand, right_hand = 1, 100
    fatigue = 0
    
    for key, hand in presses:
        if hand == 'L':
            fatigue += abs(key - left_hand)
            left_hand = key
        else:
            fatigue += abs(key - right_hand)
            right_hand = key
    
    return fatigue

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    presses = [(int(data[i]), data[i+1]) for i in range(1, 2*N, 2)]
    
    print(min_fatigue_level(N, presses))