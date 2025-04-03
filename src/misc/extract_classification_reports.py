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


def extract_reports_with_direct_pattern(log_content, debug=False):
    """Extract classification reports using direct pattern matching for the specific formats."""
    
    # Define pattern for reports with classes 0, 12, 99 (the user mentioned these classes)
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
    
    # If we found matches, convert them to structured data
    parsed_reports = []
    
    for i, match in enumerate(matches_0_12_99):
        # Look for iteration and stage information in the context before this report
        report_text = re.search(pattern_classes_0_12_99, log_content).group(0)
        report_start_idx = log_content.find(report_text)
        
        if report_start_idx >= 0:
            context_start = max(0, report_start_idx - 1000)
            context_before = log_content[context_start:report_start_idx]
            
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
        else:
            iteration = "unknown"
            stage = "unknown"
        
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
        print("First few lines:\n{0}".format(log_content[:500]))
        print("Last few lines:\n{0}".format(log_content[-500:]))
    
    # First try the direct pattern matching approach which is most likely to work
    parsed_reports = extract_reports_with_direct_pattern(log_content, debug=debug)
    
    # If we got results, return them
    if parsed_reports:
        return parsed_reports
    
    # Otherwise, try the more general approach
    if debug:
        print("Direct pattern matching failed, trying generic approach...")
    
    # Look for the "Classification report:" marker in the log file
    if "Classification report:" not in log_content:
        print("WARNING: The phrase 'Classification report:' was not found in the log file.")
        if debug:
            # Check for similar patterns that might be used instead
            for pattern in ["classification report", "Classification Report", "CLASSIFICATION REPORT"]:
                if pattern in log_content:
                    print("Found alternative pattern: '{0}'".format(pattern))
        
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