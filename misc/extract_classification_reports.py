#!/usr/bin/env python3
# Script to extract classification reports from log files and export them to CSV
import os
import re
import pandas as pd
import argparse
import glob
from pathlib import Path

# No need to update imports for this file since it doesn't import from other project modules

def parse_classification_report(report_text):
    """Parse a classification report text and convert it to a structured dictionary."""
    # Regular expressions to match different sections of the classification report
    lines = report_text.strip().split('\n')
    
    # Skip header lines
    data_lines = []
    for line in lines:
        if re.match(r'\s*\d+|accuracy|macro avg|weighted avg', line):
            data_lines.append(line)
    
    # Extract data for classes, accuracy, and averages
    metrics_data = []
    for line in data_lines:
        parts = re.split(r'\s+', line.strip())
        if len(parts) >= 5:  # For class rows: [class, precision, recall, f1-score, support]
            if parts[0] == "accuracy":
                metrics_data.append({
                    'class': 'accuracy',
                    'precision': None,
                    'recall': None,
                    'f1-score': float(parts[1]),
                    'support': int(parts[2])
                })
            else:
                metrics_data.append({
                    'class': parts[0],
                    'precision': float(parts[1]),
                    'recall': float(parts[2]),
                    'f1-score': float(parts[3]),
                    'support': int(parts[4])
                })
    
    return metrics_data


def extract_reports_with_exact_pattern(log_content, debug=False):
    """Extract classification reports using a pattern that exactly matches the format in the log file."""
    
    # Define a pattern that exactly matches the format shown in the user's example
    exact_pattern = r"""Classification report:
              precision    recall  f1-score   support

\s+0\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)
\s+12\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)
\s+99\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)

    accuracy\s+(\d+\.\d+)\s+(\d+)
   macro avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)
weighted avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)"""
    
    # Try to find matches for this exact pattern
    exact_matches = re.findall(exact_pattern, log_content, re.MULTILINE)
    
    if debug:
        print("Found {0} reports matching the exact format pattern".format(len(exact_matches)))
        
        # If we didn't find any matches, try to see if the classification report header exists
        if len(exact_matches) == 0:
            header_count = log_content.count("Classification report:")
            if header_count > 0:
                print("Found {0} 'Classification report:' headers, but couldn't match the full pattern".format(header_count))
                
                # Show a short section after one of the headers to debug
                header_pos = log_content.find("Classification report:")
                if header_pos >= 0:
                    print("Sample section after a header:")
                    print(log_content[header_pos:header_pos+300])
    
    # If we didn't find any matches with the exact pattern, try a more flexible one
    if len(exact_matches) == 0:
        # More flexible pattern that allows for variations in whitespace
        flexible_pattern = r"""Classification report:
\s+precision\s+recall\s+f1-score\s+support\s*

\s+0\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s*
\s+12\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s*
\s+99\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s*

\s+accuracy\s+(\d+\.\d+)\s+(\d+)\s*
\s+macro avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s*
\s+weighted avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)"""
        
        flexible_matches = re.findall(flexible_pattern, log_content, re.MULTILINE)
        if debug:
            print("Found {0} reports with the flexible pattern".format(len(flexible_matches)))
        
        # If we found matches with the flexible pattern, use those
        if len(flexible_matches) > 0:
            exact_matches = flexible_matches
    
    # If we still didn't find any matches, try an even more flexible pattern
    if len(exact_matches) == 0:
        # Super flexible pattern
        super_flexible_pattern = r"""Classification report:.*?
.*?precision.*?recall.*?f1-score.*?support.*?

.*?0.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+).*?
.*?12.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+).*?
.*?99.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+).*?

.*?accuracy.*?(\d+\.\d+).*?(\d+).*?
.*?macro avg.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+).*?
.*?weighted avg.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+)"""
        
        super_flexible_matches = re.findall(super_flexible_pattern, log_content, re.MULTILINE | re.DOTALL)
        if debug:
            print("Found {0} reports with super flexible pattern".format(len(super_flexible_matches)))
        
        if len(super_flexible_matches) > 0:
            exact_matches = super_flexible_matches
            
    # If we found matches, convert them to structured data
    parsed_reports = []
    
    for i, match in enumerate(exact_matches):
        # Find the position of this match in the log content
        match_text = "Classification report:"  # Start of the match
        all_occurrences = [m.start() for m in re.finditer(re.escape(match_text), log_content)]
        
        if i < len(all_occurrences):
            report_start_idx = all_occurrences[i]
            context_start = max(0, report_start_idx - 1000)
            context_before = log_content[context_start:report_start_idx]
        else:
            context_before = ""
        
        # Look for stage and iteration information
        iteration_match = re.search(r"Iteration (\d+)/\d+", context_before)
        stage_match = re.search(r"Stage (\d+)/\d+", context_before)
        
        iteration = iteration_match.group(1) if iteration_match else "unknown"
        stage = stage_match.group(1) if stage_match else "unknown"
        
        # If we couldn't find stage and iteration, look for other indicators
        if stage == "unknown" and iteration == "unknown":
            if "STAGE 1: Training initial wildlife model" in context_before:
                stage = "wildlife"
            elif "STAGE 2: Fine-tuning on possum disease data" in context_before:
                stage = "possum"
            elif "Evaluating after curriculum stage" in context_before:
                # Try to extract the stage and iteration from this pattern
                curriculum_match = re.search(r"curriculum stage (\d+) \(iteration (\d+)\)", context_before)
                if curriculum_match:
                    stage = curriculum_match.group(1)
                    iteration = curriculum_match.group(2)
        
        # Add class 0 metrics
        parsed_reports.append({
            'class': '0',
            'precision': float(match[0]),
            'recall': float(match[1]),
            'f1-score': float(match[2]),
            'support': int(match[3]),
            'iteration': iteration,
            'stage': stage,
            'report_index': i
        })
        
        # Add class 12 metrics
        parsed_reports.append({
            'class': '12',
            'precision': float(match[4]),
            'recall': float(match[5]),
            'f1-score': float(match[6]),
            'support': int(match[7]),
            'iteration': iteration,
            'stage': stage,
            'report_index': i
        })
        
        # Add class 99 metrics
        parsed_reports.append({
            'class': '99',
            'precision': float(match[8]),
            'recall': float(match[9]),
            'f1-score': float(match[10]),
            'support': int(match[11]),
            'iteration': iteration,
            'stage': stage,
            'report_index': i
        })
        
        # Add accuracy metrics
        parsed_reports.append({
            'class': 'accuracy',
            'precision': None,
            'recall': None,
            'f1-score': float(match[12]),
            'support': int(match[13]),
            'iteration': iteration,
            'stage': stage,
            'report_index': i
        })
        
        # Add macro avg metrics
        parsed_reports.append({
            'class': 'macro avg',
            'precision': float(match[14]),
            'recall': float(match[15]),
            'f1-score': float(match[16]),
            'support': int(match[17]),
            'iteration': iteration,
            'stage': stage,
            'report_index': i
        })
        
        # Add weighted avg metrics
        parsed_reports.append({
            'class': 'weighted avg',
            'precision': float(match[18]),
            'recall': float(match[19]),
            'f1-score': float(match[20]),
            'support': int(match[21]),
            'iteration': iteration,
            'stage': stage,
            'report_index': i
        })
    
    return parsed_reports


def find_and_extract_reports(log_file_path, debug=False):
    """Find all classification reports in a log file and extract them."""
    with open(log_file_path, 'r', errors='replace') as file:
        log_content = file.read()
    
    # Print some debug information about the file
    if debug:
        print("Log file size: {0} bytes".format(len(log_content)))
        print("First 100 characters:\n{0}".format(log_content[:100].replace('\n', ' ')))
        print("Last 100 characters:\n{0}".format(log_content[-100:].replace('\n', ' ')))
        
        # Check if the file contains the expected pattern
        if "Classification report:" in log_content:
            print("File contains 'Classification report:' text")
            
            # Find all occurrences of the header
            positions = [m.start() for m in re.finditer("Classification report:", log_content)]
            print("Found {0} occurrences of 'Classification report:' at positions: {1}".format(
                len(positions), positions[:5]))
            
            # Show a sample of the first report
            if positions:
                first_pos = positions[0]
                print("\nSample of first classification report:")
                print(log_content[first_pos:first_pos+500])
        else:
            print("WARNING: File does not contain 'Classification report:' text")
    
    # Try the exact matching approach first
    parsed_reports = extract_reports_with_exact_pattern(log_content, debug=debug)
    
    # If we got results, return them
    if parsed_reports:
        return parsed_reports
    
    # If still no results, try the older approach with direct pattern matching 
    if debug:
        print("\nTrying the previous direct pattern matching approach...")
    
    # Define pattern for reports with classes 0, 12, 99 
    pattern_classes_0_12_99 = r"""precision\s+recall\s+f1-score\s+support\s*
\s*0\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s*
\s*12\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s*
\s*99\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s*
\s*accuracy\s+(\d+\.\d+)\s+(\d+)\s*
\s*macro avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s*
\s*weighted avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)"""
    
    # Try to find matches for this pattern
    matches_0_12_99 = re.findall(pattern_classes_0_12_99, log_content, re.MULTILINE | re.DOTALL)
    
    if debug:
        print("Found {0} reports matching classes 0, 12, 99 pattern".format(len(matches_0_12_99)))
    
    if matches_0_12_99:
        parsed_reports = []
        for i, match in enumerate(matches_0_12_99):
            # Extract context information (simple approach)
            iteration = "unknown"
            stage = "unknown"
            
            # Add class metrics (simplified)
            for class_idx, class_name in enumerate(['0', '12', '99']):
                idx = class_idx * 4  # 4 values per class
                parsed_reports.append({
                    'class': class_name,
                    'precision': float(match[idx]),
                    'recall': float(match[idx+1]),
                    'f1-score': float(match[idx+2]),
                    'support': int(match[idx+3]),
                    'iteration': iteration,
                    'stage': stage,
                    'report_index': i
                })
            
            # Add summary metrics
            # Accuracy
            parsed_reports.append({
                'class': 'accuracy',
                'precision': None, 
                'recall': None,
                'f1-score': float(match[12]),
                'support': int(match[13]),
                'iteration': iteration, 
                'stage': stage,
                'report_index': i
            })
            
            # Macro avg
            parsed_reports.append({
                'class': 'macro avg',
                'precision': float(match[14]),
                'recall': float(match[15]),
                'f1-score': float(match[16]),
                'support': int(match[17]),
                'iteration': iteration,
                'stage': stage, 
                'report_index': i
            })
            
            # Weighted avg
            parsed_reports.append({
                'class': 'weighted avg',
                'precision': float(match[18]),
                'recall': float(match[19]),
                'f1-score': float(match[20]),
                'support': int(match[21]),
                'iteration': iteration,
                'stage': stage,
                'report_index': i
            })
        
        return parsed_reports
    
    # As a last resort, try a very simple approach
    if debug:
        print("\nTrying a basic approach to extract metrics directly...")
    
    # Look for lines with the class labels and metrics
    class_0_pattern = r"\s*0\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)"
    class_12_pattern = r"\s*12\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)"
    class_99_pattern = r"\s*99\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)"
    
    class_0_matches = re.findall(class_0_pattern, log_content)
    class_12_matches = re.findall(class_12_pattern, log_content) 
    class_99_matches = re.findall(class_99_pattern, log_content)
    
    if debug:
        print("Found direct matches for classes: 0:{0}, 12:{1}, 99:{2}".format(
            len(class_0_matches), len(class_12_matches), len(class_99_matches)))
    
    # If we have matches for all classes
    min_matches = min(len(class_0_matches), len(class_12_matches), len(class_99_matches))
    if min_matches > 0:
        basic_reports = []
        for i in range(min_matches):
            # Add each class
            basic_reports.append({
                'class': '0',
                'precision': float(class_0_matches[i][0]),
                'recall': float(class_0_matches[i][1]), 
                'f1-score': float(class_0_matches[i][2]),
                'support': int(class_0_matches[i][3]),
                'iteration': 'unknown',
                'stage': 'unknown',
                'report_index': i
            })
            
            basic_reports.append({
                'class': '12',
                'precision': float(class_12_matches[i][0]),
                'recall': float(class_12_matches[i][1]),
                'f1-score': float(class_12_matches[i][2]),
                'support': int(class_12_matches[i][3]),
                'iteration': 'unknown',
                'stage': 'unknown',
                'report_index': i
            })
            
            basic_reports.append({
                'class': '99',
                'precision': float(class_99_matches[i][0]),
                'recall': float(class_99_matches[i][1]),
                'f1-score': float(class_99_matches[i][2]), 
                'support': int(class_99_matches[i][3]),
                'iteration': 'unknown',
                'stage': 'unknown',
                'report_index': i
            })
        
        return basic_reports
            
    # If all else fails, try the general approach
    if debug:
        print("\nAll specific approaches failed. Trying generic approach...")
    
    # Try multiple patterns to match classification reports
    # Pattern 1: Standard format with newlines
    report_pattern1 = r"Classification report:\s*\n(.*?)(?=\n\n|\Z)"
    # Pattern 2: More flexible spacing
    report_pattern2 = r"Classification report:[ \t]*(.*?)(?=\n\n|\Z)"
    # Pattern 3: Match anything after "Classification report:" until we hit a blank line
    report_pattern3 = r"Classification report:[^\n]*\n((?:.*\n)+?)(?:\s*\n|\Z)"
    
    all_patterns = [report_pattern1, report_pattern2, report_pattern3]
    
    all_reports = []
    for i, pattern in enumerate(all_patterns):
        reports = re.findall(pattern, log_content, re.DOTALL | re.IGNORECASE)
        if reports:
            print("Found {0} reports using pattern {1}".format(len(reports), i+1))
            all_reports.extend(reports)
            # If we found reports, no need to try other patterns
            break
    
    if not all_reports:
        print("WARNING: Could not extract any classification reports with any pattern.")
        # Try to find if there's any content that looks like a classification report
        # Look for lines with precision, recall and f1-score
        possible_reports = re.findall(r"(precision\s+recall\s+f1-score.*(?:\n.*){3,15})", log_content, re.DOTALL | re.IGNORECASE)
        if possible_reports:
            print("Found {0} possible report-like sections by looking for precision/recall headers.".format(len(possible_reports)))
            # Use these as fallback
            all_reports = possible_reports
    
    # Parse each report
    parsed_reports = []
    for i, report in enumerate(all_reports):
        try:
            # For debugging
            if debug and i < 2:  # Just show the first two reports
                print("\nReport {0}:\n{1}".format(i+1, report))
            
            # Extract the stage/iteration information from nearby lines
            # Look for context within 1000 characters before the report
            report_start_idx = log_content.find(report)
            if report_start_idx >= 0:
                context_start = max(0, report_start_idx - 1000)
                context_before = log_content[context_start:report_start_idx]
            else:
                # If we can't find the exact position (unlikely), just use a split approach
                context_before = log_content.split("Classification report:")[0][-1000:]
            
            # Look for stage and iteration information
            iteration_match = re.search(r"Iteration (\d+)/\d+", context_before)
            stage_match = re.search(r"Stage (\d+)/\d+", context_before)
            
            iteration = iteration_match.group(1) if iteration_match else "unknown"
            stage = stage_match.group(1) if stage_match else "unknown"
            
            # If we couldn't find stage and iteration, look for other indicators
            if stage == "unknown" and iteration == "unknown":
                if "STAGE 1: Training initial wildlife model" in context_before:
                    stage = "wildlife"
                elif "STAGE 2: Fine-tuning on possum disease data" in context_before:
                    stage = "possum"
            
            # Parse the report
            parsed_report = parse_classification_report(report)
            
            # Add stage/iteration information to each row
            for row in parsed_report:
                row['iteration'] = iteration
                row['stage'] = stage
                row['report_index'] = i
            
            parsed_reports.extend(parsed_report)
        except Exception as e:
            print("Error parsing report {0}: {1}".format(i, e))
            if debug:
                print("Problematic report content:\n{0}".format(report[:200]))
            continue
    
    return parsed_reports


def export_reports_to_csv(parsed_reports, output_dir, prefix="classification_reports"):
    """Export the parsed reports to CSV files."""
    if not parsed_reports:
        print("No reports found to export.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(parsed_reports)
    
    # Save all reports to a single CSV
    all_reports_path = os.path.join(output_dir, "{0}_all.csv".format(prefix))
    df.to_csv(all_reports_path, index=False)
    print("Exported all classification reports to: {0}".format(all_reports_path))
    
    # Export separate CSVs for each iteration and stage
    # Group by iteration and stage
    for (iteration, stage), group_df in df.groupby(['iteration', 'stage']):
        # Skip groups with unknown iteration or stage if needed
        if iteration == "unknown" and stage == "unknown":
            continue
            
        # Create filename - replace f-string with format()
        if iteration != "unknown":
            filename = "{0}_iteration_{1}_stage_{2}.csv".format(prefix, iteration, stage)
        else:
            filename = "{0}_stage_{1}.csv".format(prefix, stage)
        
        # Export CSV
        csv_path = os.path.join(output_dir, filename)
        group_df.to_csv(csv_path, index=False)
        print("Exported report for iteration {0}, stage {1} to: {2}".format(
            iteration, stage, csv_path))
    
    # Also create a summary DataFrame with key metrics
    if 'class' in df.columns and 'f1-score' in df.columns:
        try:
            # Get metrics for each report (identified by report_index)
            summaries = []
            for idx, group in df.groupby('report_index'):
                # Create summary row
                summary = {
                    'iteration': group['iteration'].iloc[0],
                    'stage': group['stage'].iloc[0],
                    'report_index': idx
                }
                
                # Add metrics for each class
                for _, row in group.iterrows():
                    class_name = row['class']
                    if class_name in ['0', '12', '99']:
                        class_prefix = "class_{0}_".format(class_name)
                    else:
                        class_prefix = "{0}_".format(class_name.replace(' ', '_'))
                    
                    # Add precision, recall, f1-score for this class
                    if row['precision'] is not None:
                        summary[class_prefix + 'precision'] = row['precision']
                    if row['recall'] is not None:
                        summary[class_prefix + 'recall'] = row['recall']
                    summary[class_prefix + 'f1-score'] = row['f1-score']
                    summary[class_prefix + 'support'] = row['support']
                
                summaries.append(summary)
            
            # Create and save summary DataFrame
            if summaries:
                summary_df = pd.DataFrame(summaries)
                summary_path = os.path.join(output_dir, "{0}_summary.csv".format(prefix))
                summary_df.to_csv(summary_path, index=False)
                print("Exported summary metrics to: {0}".format(summary_path))
        except Exception as e:
            print("Error creating summary DataFrame: {0}".format(e))
    
    return all_reports_path


def main():
    parser = argparse.ArgumentParser(description='Extract classification reports from log files')
    parser.add_argument('--log_file', type=str, help='Path to the log file containing classification reports')
    parser.add_argument('--log_dir', type=str, help='Directory containing log files to process')
    parser.add_argument('--output_dir', type=str, default='classification_reports', 
                        help='Directory to save the extracted reports (default: classification_reports)')
    parser.add_argument('--prefix', type=str, default='classification_report',
                        help='Prefix for the CSV filenames (default: classification_report)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode to print additional information')
    
    args = parser.parse_args()
    
    if not args.log_file and not args.log_dir:
        parser.error("Either --log_file or --log_dir must be specified")
    
    all_parsed_reports = []
    
    # Process single log file
    if args.log_file:
        if not os.path.exists(args.log_file):
            parser.error("Log file not found: {0}".format(args.log_file))
        print("Processing log file: {0}".format(args.log_file))
        parsed_reports = find_and_extract_reports(args.log_file, debug=args.debug)
        all_parsed_reports.extend(parsed_reports)
    
    # Process all log files in directory
    if args.log_dir:
        if not os.path.exists(args.log_dir):
            parser.error("Log directory not found: {0}".format(args.log_dir))
        
        log_files = glob.glob(os.path.join(args.log_dir, "*.log")) + glob.glob(os.path.join(args.log_dir, "*.txt"))
        if not log_files:
            parser.error("No log files found in directory: {0}".format(args.log_dir))
        
        for log_file in log_files:
            print("Processing log file: {0}".format(log_file))
            parsed_reports = find_and_extract_reports(log_file, debug=args.debug)
            all_parsed_reports.extend(parsed_reports)
    
    # Export all parsed reports
    if all_parsed_reports:
        print("Found {0} report entries in total".format(len(all_parsed_reports)))
        export_reports_to_csv(all_parsed_reports, args.output_dir, args.prefix)
    else:
        print("No classification reports found in the provided log file(s)")


if __name__ == "__main__":
    main() 