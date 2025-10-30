import sys
import itertools

def solve(N, X, Y, dishes):
    max_dishes = 0
    for order in itertools.permutations(dishes):
        sweetness = 0
        saltiness = 0
        dishes_eaten = 0
        for dish in order:
            if sweetness + dish[0] <= X and saltiness + dish[1] <= Y:
                sweetness += dish[0]
                saltiness += dish[1]
                dishes_eaten += 1
            else:
                break
        max_dishes = max(max_dishes, dishes_eaten)
    return max_dishes

def main():
    N, X, Y = map(int, sys.stdin.readline().split())
    dishes = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    print(solve(N, X, Y, dishes))

if __name__ == "__main__":
    main()