import sys

def max_tastiness(N, courses):
    # Initialize variables to keep track of the maximum tastiness
    healthy_max = 0
    upset_max = 0
    current_healthy = 0
    current_upset = 0

    for X, Y in courses:
        if X == 0:  # Antidotal course
            current_healthy += Y
            current_upset += Y
        else:  # Poisonous course
            # If we eat the poisonous course with a healthy stomach
            new_upset = current_healthy + Y
            # If we skip the poisonous course with a healthy stomach
            new_healthy = current_healthy
            # Update the maximum tastiness for healthy stomach
            healthy_max = max(healthy_max, current_healthy)
            # Update the current tastiness for upset stomach
            current_upset = max(current_upset, new_upset)
            # Update the current tastiness for healthy stomach
            current_healthy = new_healthy

        # Update the maximum tastiness for upset stomach
        upset_max = max(upset_max, current_upset)

    # The final answer is the maximum of the healthy and upset stomach tastiness
    return max(healthy_max, upset_max)

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    courses = []
    index = 1
    for i in range(N):
        X = int(data[index])
        Y = int(data[index + 1])
        courses.append((X, Y))
        index += 2

    result = max_tastiness(N, courses)
    print(result)

if __name__ == "__main__":
    main()