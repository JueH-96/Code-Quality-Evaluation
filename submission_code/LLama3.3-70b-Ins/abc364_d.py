import sys

def calculate_distance(a, b):
    """Calculate the distance between two points on a number line."""
    return abs(a - b)

def find_kth_closest_point(a, b, k):
    """Find the kth closest point to point b among points a."""
    distances = [calculate_distance(x, b) for x in a]
    sorted_distances = sorted(distances)
    kth_distance = sorted_distances[k - 1]
    return kth_distance

def main():
    """Read input from stdin and write output to stdout."""
    n, q = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    for _ in range(q):
        b, k = map(int, sys.stdin.readline().split())
        kth_distance = find_kth_closest_point(a, b, k)
        print(kth_distance)

if __name__ == "__main__":
    main()