function AutoData2()
    if ~isfolder('training_data')
        mkdir('training_data');
    end
    if ~isfolder('validation_data')
        mkdir('validation_data');
    end
    if ~isfolder('testing_data')
        mkdir('testing_data');
    end

    % Global run multiplier
    global_run_multiplier = 25;
    local_run_multiplier = 100;

    % --- Training Data ---
    disp('Generating Training Data...');
    diameters = [40, 55, 70, 85, 100];
    lengths = [30, 60, 90, 120, 150];
    run_count = 1;
    for d = diameters
        for l = lengths
            for run = 1:(2 * global_run_multiplier)
                rng(run_count);
                filename = sprintf('training_data/cable_data_D%d_L%d_run%d.csv', d, l, run);
                cable_integrity_simulator_v17_modified(d, l, 10, 240, 12, run_count, filename);
                run_count = run_count + 1;
            end
        end
    end
    extra_combinations = [
        70, 90, 2 * global_run_multiplier;
        40, 30, global_run_multiplier;
        40, 150, global_run_multiplier;
        100, 30, global_run_multiplier;
        100, 150, global_run_multiplier;
        70, 90, global_run_multiplier;
    ];
    for i = 1:size(extra_combinations, 1)
        d = extra_combinations(i, 1);
        l = extra_combinations(i, 2);
        num_runs = extra_combinations(i, 3);
        for run = 1:num_runs
            rng(run_count);
            filename = sprintf('training_data/cable_data_D%d_L%d_run%d.csv', d, l, run_count);
            cable_integrity_simulator_v17_modified(d, l, 10, 240, 12, run_count, filename);
            run_count = run_count + 1;
        end
    end
    disp('Training Data Complete.');

    % --- Validation Data ---
    disp('Generating Validation Data...');
    diameters = [45, 70, 95];
    lengths = [45, 90, 135];
    run_count = 1;
    for d = diameters
        for l = lengths
            for run = 1:global_run_multiplier
                rng(run_count);
                filename = sprintf('validation_data/cable_data_D%d_L%d_run%d.csv', d, l, run);
                cable_integrity_simulator_v17_modified(d, l, 10, 240, 12, run_count, filename);
                run_count = run_count + 1;
            end
        end
    end
     extra_combinations = [
        45, 90, global_run_multiplier;
        70, 45, global_run_multiplier;
        95, 135, global_run_multiplier;
    ];
    for i = 1:size(extra_combinations, 1)
        d = extra_combinations(i, 1);
        l = extra_combinations(i, 2);
        num_runs = extra_combinations(i, 3);
        for run = 1:num_runs
            rng(run_count);
            filename = sprintf('validation_data/cable_data_D%d_L%d_run%d.csv', d, l, run_count);
            cable_integrity_simulator_v17_modified(d, l, 10, 240, 12, run_count, filename);
            run_count = run_count + 1;
        end
    end
    disp('Validation Data Complete.');

    % --- Testing Data ---
    disp('Generating Testing Data...');
    diameters = [50, 80];
    lengths = [50, 110];
    run_count = 1;
    for d = diameters
        for l = lengths
            for run = 1:local_run_multiplier
                rng(run_count);
                filename = sprintf('testing_data/cable_data_D%d_L%d_run%d.csv', d, l, run);
                cable_integrity_simulator_v17_modified(d, l, 10, 240, 12, run_count, filename);
                run_count = run_count + 1;
            end
        end
    end
     extra_combinations = [
        50, 50, local_run_multiplier;
        80, 110, local_run_multiplier;
    ];
    for i = 1:size(extra_combinations, 1)
        d = extra_combinations(i, 1);
        l = extra_combinations(i, 2);
        num_runs = extra_combinations(i, 3);
        for run = 1:num_runs
            rng(run_count);
            filename = sprintf('testing_data/cable_data_D%d_L%d_run%d.csv', d, l, run_count);
            cable_integrity_simulator_v17_modified(d, l, 10, 240, 12, run_count, filename);
            run_count = run_count + 1;
        end
    end
    disp('Testing Data Complete.');
end


% --- Simulator function ---
function cable_integrity_simulator_v17_modified(cableDiameter, cableLength, durationYears, sampleIntervalMinutes, numCables, seed, filename)

    % --- Input Validation ---
    if any([cableDiameter, cableLength, numCables, durationYears, sampleIntervalMinutes] <= 0) || ...
       any(isnan([cableDiameter, cableLength, numCables, durationYears, sampleIntervalMinutes]))
        error('All numeric inputs must be positive and non-NaN.');
    end
    if numCables ~= floor(numCables)
        error('Number of cables must be an integer.');
    end
    if ~isnumeric(seed) || isnan(seed) || ~isfinite(seed)
        error('Seed must be a numeric, finite, and non-NaN value.');
    end
    if ~ischar(filename) && ~isstring(filename)
       error('Filename must be a string.');
    end

    % --- Parameter Setup ---
    bridge_params = struct('cable_diameter_mm', cableDiameter, 'cable_length_m', cableLength, 'num_cables', numCables);
    sim_params = struct('duration_years', durationYears, 'duration_months', durationYears * 12, ...
                        'sampling_interval_minutes', sampleIntervalMinutes);

    % --- Break Amplitude Parameters ---
    break_amplitude_base = [0.3, 0.4];

    % --- Calculate Time and Samples ---
    start_date = datetime('now');
    end_date = start_date + years(sim_params.duration_years);
    total_minutes = minutes(end_date - start_date);
    num_samples = floor(total_minutes / sim_params.sampling_interval_minutes) + 1;

    % --- Input validation for num_samples ---
    if ~isnumeric(num_samples) || isnan(num_samples) || ~isfinite(num_samples) || num_samples <=0
        error('Calculated number of samples is invalid. Check durationYears and sampleIntervalMinutes.');
    end

    % --- Initialize Data Matrix ---
    cable_data_matrix = zeros(num_samples, bridge_params.num_cables);
    cable_status_labels = zeros(1, bridge_params.num_cables);

    % --- Filter Tuning ---
     filter_params = struct(...
        'x1',  struct('movmean_hours', 5, 'sgolay_hours', 2), ...
        'x5',  struct('movmean_hours', 15,   'sgolay_hours', 5), ...
        'x10', struct('movmean_hours', 2,   'sgolay_hours', 0.8), ...
        'x30', struct('movmean_hours', 24,   'sgolay_hours', 10), ...
        'x240', struct('movmean_hours', 30,   'sgolay_hours', 12)  ...
    );
    interval_key = ['x', num2str(sim_params.sampling_interval_minutes)];
    if ~isfield(filter_params, interval_key)
        error('Filter parameters not defined for the selected sampling interval.');
    end
    current_filter_params = filter_params.(interval_key);
    movmean_window_samples = round(current_filter_params.movmean_hours * 60 / sim_params.sampling_interval_minutes);
    sgolay_window_samples = round(current_filter_params.sgolay_hours * 60 / sim_params.sampling_interval_minutes);
    sgolay_window_samples = sgolay_window_samples + (1 - mod(sgolay_window_samples,2));
    sgolay_window_samples = max(sgolay_window_samples, 5);

    % --- Bridge Status (0-9:  Class labels) ---
    bridge_status = randi([0, 9]);
    bridge_breakage_modifier = 1 + (bridge_status * (0.4 / 9));

    % --- Simulate Data for each Cable ---
    for cable_index = 1:bridge_params.num_cables
        cable_signal = zeros(num_samples, 1);

        % --- Cable Status (0-9) ---
        cable_status = randi([0, 9]);
        cable_status_labels(cable_index) = cable_status;

        % --- Breakage Parameters based on Status ---
        base_rates = linspace(0.05, 0.4, 10);
        accumulated_increases = linspace(0.001, 0.02, 10);

        base_monthly_breakage_rate = base_rates(cable_status + 1);
        accumulated_breakage_prob_increase = accumulated_increases(cable_status + 1);
        accumulated_breakage_probability = base_monthly_breakage_rate;

        for month = 1:sim_params.duration_months
            % --- Breakage Simulation ---
            if rand() < (accumulated_breakage_probability * bridge_breakage_modifier)
                num_breaks_this_month = poissrnd(base_monthly_breakage_rate*4);
                for break_num = 1:num_breaks_this_month
                    break_time_index = randi([max(1, round(((month-1)*30*24*60) / sim_params.sampling_interval_minutes)), ...
                                             min(num_samples, round((month*30*24*60) / sim_params.sampling_interval_minutes))]);
                    if break_time_index > 0 && break_time_index <= num_samples
                        amplitude =  break_amplitude_base(1) + (break_amplitude_base(2) - break_amplitude_base(1)) * rand();
                        cable_signal(break_time_index) = cable_signal(break_time_index) + amplitude;
                    end
                end
            end

            % --- Accumulated Breakage Probability ---
            accumulated_breakage_probability = accumulated_breakage_probability + ...
                                               accumulated_breakage_prob_increase * month ;
            accumulated_breakage_probability = min(accumulated_breakage_probability, 1.0);
        end

        % --- Generate and Add Noise ---
        noise_signal = 0.005 * randn(num_samples, 1);
        cable_signal = cable_signal + noise_signal;

        % --- Apply Filtering ---
        filtered_cable_signal = movmean(cable_signal, movmean_window_samples);
        filtered_cable_signal = sgolayfilt(filtered_cable_signal, 3, sgolay_window_samples);
        cable_data_matrix(:, cable_index) = filtered_cable_signal;
    end

    % --- Display and Save Results ---
    column_names = cellstr("Cable_" + string(1:bridge_params.num_cables));
     output_table = array2table(cable_data_matrix, 'VariableNames', column_names);
     status_row = array2table(cable_status_labels, 'VariableNames', column_names);
     output_table = [output_table; status_row];
     writetable(output_table, filename);
end