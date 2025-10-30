def solve():
    # Read N, the number of people
    N = int(input())

    # Store data for each person
    # all_people_data will be a list of dictionaries.
    # Each dictionary: {'id': person_id, 'num_bets': C_i, 'bets': set_of_bets}
    all_people_data = []
    for i in range(N):
        person_id = i + 1  # Person IDs are 1-indexed

        C_i = int(input())  # Number of bets for this person
        
        # Read bets as a list of strings, then map to integers and convert to a set
        bets_str_list = input().split()
        bets_set = set(map(int, bets_str_list))
        
        all_people_data.append({
            'id': person_id,
            'num_bets': C_i,
            'bets': bets_set
        })

    # Read X, the winning number
    X = int(input())

    # Filter to find people who bet on X
    # people_who_bet_on_X will be a list of dictionaries.
    # Each dictionary: {'id': person_id, 'num_bets': C_i}
    people_who_bet_on_X = []
    for person_data in all_people_data:
        if X in person_data['bets']:
            people_who_bet_on_X.append({
                'id': person_data['id'],
                'num_bets': person_data['num_bets']
            })

    # This list will store the IDs of the final selected people
    output_ids = []

    if not people_who_bet_on_X:
        # No one bet on X. output_ids remains empty.
        pass
    else:
        # At least one person bet on X.
        # Find the minimum number of bets among them.
        min_num_bets = float('inf')
        for p_data in people_who_bet_on_X:
            if p_data['num_bets'] < min_num_bets:
                min_num_bets = p_data['num_bets']
        
        # Collect IDs of people who bet on X and have this minimum number of bets.
        for p_data in people_who_bet_on_X:
            if p_data['num_bets'] == min_num_bets:
                output_ids.append(p_data['id'])
        
        # Sort the collected IDs in ascending order.
        output_ids.sort()

    # Print the results
    # First line: count of selected people (K)
    print(len(output_ids))
    
    # Second line: selected person IDs (B_1, B_2, ..., B_K), space-separated.
    # If output_ids is empty, print(*output_ids) will print a newline,
    # resulting in an empty second line, which matches typical contest output formats.
    print(*output_ids)

if __name__ == '__main__':
    solve()