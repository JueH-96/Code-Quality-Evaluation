def main():
    # Read N, T, P from the first line of input
    N, T, P = map(int, input().split())

    # Read the list of hair lengths L from the second line of input
    L = list(map(int, input().split()))

    # For each person, calculate the number of days they need for their hair
    # to reach length T.
    # If a person's hair length L_i is already T or greater, they need 0 days.
    # Otherwise, they need T - L_i days.
    # This can be calculated as max(0, T - L_i).
    days_needed_for_each_person = []
    for hair_length in L:
        days = T - hair_length
        if days < 0:  # Hair is already T or longer
            days_needed_for_each_person.append(0)
        else:
            days_needed_for_each_person.append(days)
    
    # A more Pythonic way to achieve the above list construction:
    # days_needed_for_each_person = [max(0, T - hair_length) for hair_length in L]

    # Sort these "days needed" values in non-decreasing order.
    days_needed_for_each_person.sort()

    # We need P people to have hair length of at least T.
    # The P-th value in the sorted list (which is at index P-1 due to 0-indexing)
    # represents the minimum number of days required for this condition to be met.
    # After this many days, the P people who reach length T the fastest
    # (or were already there) will all satisfy the hair length condition.
    # If we waited one day less (and this P-1th value is > 0), the P-th person
    # in this sorted list would not yet have hair of length T,
    # meaning fewer than P people would satisfy the condition.
    # Thus, days_needed_for_each_person[P-1] is the number of days
    # after which the condition (P or more people with hair length >= T)
    # is met for the first time.
    result_days = days_needed_for_each_person[P-1]

    # Print the result.
    print(result_days)

if __name__ == '__main__':
    main()