from typing import List

def calculate_max_difference(boxes: List[int]) -> int:
    """
    Calculates the maximum absolute difference between the total weights of any two trucks.
    
    Args:
    boxes: A list of integers representing the weights of the boxes.
    
    Returns:
    An integer representing the maximum absolute difference between the total weights of any two trucks.
    """
    total_weight = sum(boxes)
    max_diff = 0
    for k in range(1, len(boxes) + 1):
        if len(boxes) % k != 0:
            continue
        groups = [0] * (len(boxes) // k)
        for i in range(len(boxes)):
            groups[i // k] += boxes[i]
        max_diff = max(max_diff, max(groups) - min(groups))
    return max_diff

def main():
    t = int(input())
    for _ in range(t):
        n = int(input())
        a = list(map(int, input().split()))
        print(calculate_max_difference(a))

if __name__ == "__main__":
    main()