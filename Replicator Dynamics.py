# Replicator dynamics
import numpy as np
number_of_iterations = 100
for i in range(number_of_iterations):
    for player in range(0, len(words)):
        player_payoff = 0
        strategy_payoff = np.zeros((sense_count[player], 1))
        for neighbour in range(0, len(words)):
            if player == neighbour:
                continue
            payoff_matrix = sense_similarity_matrix[senses_start_index[player]:senses_start_index[player]+sense_count[player]+1, senses_start_index[neighbour]:senses_start_index[neighbour]+sense_count[neighbour]+1]
            sense_preference_neighbour = strategy_space[neighbour][senses_start_index[neighbour]:senses_start_index[neighbour]+sense_count[neighbour]+1]
            np.reshape(sense_preference_neighbour, (sense_count[neighbour], 1))
            sense_preference_player = strategy_space[player][senses_start_index[player]:senses_start_index[player] + sense_count[player] + 1]
            np.reshape(sense_preference_player, (sense_count[player], 1))
            current_payoff = word_similarity_matrix[player][neighbour] * np.dot(payoff_matrix, sense_preference_neighbour)
            strategy_payoff = np.add(current_payoff, strategy_payoff)
            player_payoff = np.dot(sense_preference_player.T, current_payoff) + player_payoff
        updation_values = np.divide(strategy_payoff, player_payoff)
        for j in range(0, sense_count[player]):
            strategy_space[player][senses_start_index[player]+j] = strategy_space[player][senses_start_index[player]+j] * updation_values[j]

print(strategy_space)