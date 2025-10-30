import math
import itertools

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = int(data[1])
    T = int(data[2])
    
    segments = []
    idx = 3
    for _ in range(N):
        A = int(data[idx])
        B = int(data[idx+1])
        C = int(data[idx+2])
        D = int(data[idx+3])
        segments.append( ((A, B), (C, D)) )
        idx += 4
    
    lengths = [math.hypot(segments[i][0][0] - segments[i][1][0], segments[i][0][1] - segments[i][1][1]) for i in range(N)]
    
    perms = list(itertools.permutations(range(N)))
    start_choices = list(itertools.product([0, 1], repeat=N))
    
    min_time = float('inf')
    
    for perm in perms:
        for choices in start_choices:
            current_position = (0, 0)
            total_time = 0.0
            for idx in perm:
                starting_endpoint = segments[idx][choices[idx]]
                ending_endpoint = segments[idx][1 - choices[idx]]
                if current_position == starting_endpoint:
                    movement_time = 0.0
                else:
                    movement_time = math.hypot(starting_endpoint[0] - current_position[0], starting_endpoint[1] - current_position[1]) / S
                total_time += movement_time
                print_time = lengths[idx] / T
                total_time += print_time
                current_position = ending_endpoint
            if total_time < min_time:
                min_time = total_time
    
    print("{0:.10f}".format(min_time))

if __name__ == "__main__":
    main()