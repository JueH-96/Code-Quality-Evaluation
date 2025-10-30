class Solution:
    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        mentions = [0] * numberOfUsers
        offline_until = [0] * numberOfUsers  # Initialize all users to online
        
        # Prepare events for sorting with their original indices
        sorted_events = []
        for idx, event in enumerate(events):
            timestamp = int(event[1])
            type_order = 0 if event[0] == 'OFFLINE' else 1
            sorted_events.append((timestamp, type_order, idx, event))
        
        # Sort the events based on timestamp, type_order, and original index
        sorted_events.sort(key=lambda x: (x[0], x[1], x[2]))
        ordered_events = [event[3] for event in sorted_events]
        
        # Process each event in the sorted order
        for event in ordered_events:
            current_time = int(event[1])
            if event[0] == 'OFFLINE':
                user_id = int(event[2])
                offline_until[user_id] = current_time + 60
            else:
                # Process MESSAGE event
                mentions_string = event[2]
                tokens = mentions_string.split()
                mentioned_ids = []
                
                for token in tokens:
                    if token == 'ALL':
                        # Mention all users
                        mentioned_ids.extend(range(numberOfUsers))
                    elif token == 'HERE':
                        # Mention online users
                        for u in range(numberOfUsers):
                            if current_time >= offline_until[u]:
                                mentioned_ids.append(u)
                    elif token.startswith('id'):
                        # Parse individual user ID
                        s = token[2:]
                        if s.isdigit():
                            u = int(s)
                            mentioned_ids.append(u)
                
                # Update mentions count
                for u in mentioned_ids:
                    mentions[u] += 1
        
        return mentions