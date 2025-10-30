# YOUR CODE HERE

import sys

def min_fatigue():
    n = int(sys.stdin.readline())
    keys = [0]*101
    left_hand, right_hand = 0, 0
    for _ in range(n):
        a, s = sys.stdin.readline().split()
        a = int(a)
        if s == 'L':
            if left_hand == 0:
                left_hand = a
            else:
                keys[left_hand] += abs(a - left_hand)
                left_hand = a
        else:
            if right_hand == 0:
                right_hand = a
            else:
                keys[right_hand] += abs(a - right_hand)
                right_hand = a
    keys[left_hand] += abs(left_hand - right_hand)
    keys[right_hand] += abs(left_hand - right_hand)
    print(sum(keys))

min_fatigue()