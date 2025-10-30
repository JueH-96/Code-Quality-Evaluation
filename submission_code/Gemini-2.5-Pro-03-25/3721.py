import collections # Import collections (though not used) for standard practice
from typing import List # Import List for type hinting

class Solution:
    """
    Solves the user mention counting problem based on event logs.

    Processes events chronologically, updating user online/offline status
    and counting mentions according to message types (ALL, HERE, id<number>).
    Handles the rule that status changes (offline events) are processed 
    before message events at the same timestamp by using a custom sort key.
    """
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        """
        Counts mentions for each user based on a list of events.

        Args:
            numberOfUsers: The total number of users.
            events: A list of events, where each event is
                    ["MESSAGE", timestamp_str, mentions_string] or
                    ["OFFLINE", timestamp_str, user_id_str]. 
                    Timestamps and user IDs are provided as strings.

        Returns:
            A list where the i-th element is the total mention count for user i.
        """

        # Define a sort key function.
        # Sorts primarily by timestamp (integer value).
        # Secondarily, ensures OFFLINE events come before MESSAGE events
        # at the same timestamp (OFFLINE priority 0, MESSAGE priority 1).
        # This correctly implements the rule: "status change is processed 
        # before handling any message event that occurs at the same timestamp".
        def sort_key(event):
            # event[1] is the timestamp string. Convert to int for sorting.
            # Constraints guarantee timestamp is a valid integer string between 1 and 10^5.
            timestamp = int(event[1]) 
            
            # Assign priority: 0 for OFFLINE, 1 for MESSAGE
            # This ensures OFFLINE events at a given timestamp are processed first.
            priority = 0 if event[0] == "OFFLINE" else 1
            return (timestamp, priority)

        # Sort events in place using the custom key
        events.sort(key=sort_key)

        # Initialize mention counts for each user to zero.
        mentions = [0] * numberOfUsers
        
        # online_again_at[user_id] stores the timestamp when the user
        # becomes online again after going offline.
        # A value of 0 indicates the user is currently online. This includes
        # the initial state and the state after their offline duration has passed.
        online_again_at = [0] * numberOfUsers # All users start online

        # Process events in the sorted order
        for event in events:
            event_type = event[0]
            # Convert timestamp string to integer for time comparisons.
            current_ts = int(event[1]) 
            # Data depends on event type (mentions string or user ID string).
            data = event[2] 

            if event_type == "OFFLINE":
                try:
                    # Convert user ID string to integer.
                    user_id = int(data)
                    # Validate user_id range (though constraints guarantee 0 <= user_id < numberOfUsers)
                    if 0 <= user_id < numberOfUsers:
                        # The user goes offline now. They will be automatically online again
                        # at current_ts + 60. Record this time.
                        # Constraint guarantees the user is online just before this event.
                        online_again_at[user_id] = current_ts + 60
                except ValueError:
                    # This case should not occur based on constraints but handles potential invalid user ID format.
                    pass # Ignore malformed OFFLINE events if necessary

            elif event_type == "MESSAGE":
                mentions_string = data
                
                if mentions_string == "ALL":
                    # Mention every user, regardless of their online status.
                    for user_id in range(numberOfUsers):
                        mentions[user_id] += 1
                elif mentions_string == "HERE":
                    # Mention only users who are currently online at current_ts.
                    for user_id in range(numberOfUsers):
                        # Check online status: A user is online if the current time (current_ts)
                        # is greater than or equal to the time they are scheduled to become online 
                        # again (online_again_at[user_id]). If online_again_at[user_id] is 0, 
                        # it means they are currently online (initial state or recovered).
                        is_online = (current_ts >= online_again_at[user_id])
                        if is_online:
                            mentions[user_id] += 1
                else: 
                    # Mention specific users by ID: "id<number> id<number> ..."
                    # Split the string by single whitespace.
                    tokens = mentions_string.split(' ')
                    for token in tokens:
                        # Check if token starts with "id" and has characters following "id".
                        if token.startswith("id") and len(token) > 2: 
                            try:
                                # Extract the number part after "id".
                                user_id_str = token[2:]
                                user_id = int(user_id_str)
                                # Validate the parsed user ID range (0 <= user_id < numberOfUsers)
                                if 0 <= user_id < numberOfUsers:
                                    # Mention the user regardless of their online status.
                                    # Duplicate mentions in the string are counted separately.
                                    mentions[user_id] += 1
                            except ValueError:
                                # Ignore tokens where the part after "id" is not a valid integer
                                # (e.g., "idabc", "id-1"). Constraints likely prevent this.
                                pass
        
        # Return the final mention counts for all users.
        return mentions