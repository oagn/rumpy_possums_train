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
        # This regex will match class rows (numbers) as well as summary rows
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
                # Class rows (might be any numeric class)
                metrics_data.append({
                    'class': parts[0],
                    'precision': float(parts[1]),
                    'recall': float(parts[2]),
                    'f1-score': float(parts[3]),
                    'support': int(parts[4])
                })
        elif "avg" in line and len(parts) >= 5:  # For average rows
            metrics_data.append({
                'class': ' '.join(parts[:-4]).strip(),  # Join words before metrics
                'precision': float(parts[-4]),
                'recall': float(parts[-3]),
                'f1-score': float(parts[-2]),
                'support': int(parts[-1])
            })
    
    return metrics_data


def extract_reports_with_flexible_pattern(log_content, debug=False):
    """Extract classification reports using a flexible pattern that works with any class set."""
    
    # Find all "Classification report:" headers
    report_header_positions = [m.start() for m in re.finditer(r"Classification report:", log_content)]
    
    if debug:
        print(f"Found {len(report_header_positions)} 'Classification report:' headers")
        if report_header_positions:
            for i, pos in enumerate(report_header_positions[:3]):  # Show first 3 for debugging
                print(f"Header {i+1} at position {pos}:")
                print(log_content[pos:pos+200].replace('\n', ' ')[:100] + "...")
    
    # If no headers found, return empty list
    if not report_header_positions:
        return []
    
    parsed_reports = []
    
    for i, pos in enumerate(report_header_positions):
        # Extract a chunk of text after the header (enough to contain a full report)
        chunk_end = pos + 4000  # Increased from 2000 to 4000 to ensure we get the full report
        if i + 1 < len(report_header_positions) and report_header_positions[i + 1] < chunk_end:
            chunk_end = report_header_positions[i + 1]  # Stop at the next report header
        
        chunk = log_content[pos:chunk_end]
        
        if debug and i < 2:  # Show first 2 chunks for debugging
            print(f"\nChunk {i+1} (first 300 chars):")
            print(chunk[:300].replace('\n', ' '))
        
        # Find the header line with "precision recall f1-score support"
        header_match = re.search(r"precision\s+recall\s+f1-score\s+support", chunk)
        if not header_match:
            if debug:
                print(f"Couldn't find header line in report {i+1}")
            continue
        
        # Get the part starting from the header line
        report_start = header_match.start()
        report_chunk = chunk[report_start:]
        
        if debug and i < 2:
            print(f"\nReport chunk {i+1} (first 300 chars after header):")
            print(report_chunk[:300].replace('\n', ' '))
        
        # Find where the report ends (usually two blank lines or end of chunk)
        report_end_match = re.search(r"\n\s*\n", report_chunk)
        if report_end_match:
            report_chunk = report_chunk[:report_end_match.end()]
        
        # Super flexible approach to extract class rows
        # This matches any row that starts with digits followed by decimals
        class_rows = []
        
        # First try the standard pattern for lines with class number + 4 numeric values
        std_rows = re.findall(r"\s*(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s*", report_chunk)
        if std_rows:
            class_rows = std_rows
            if debug:
                print(f"Found {len(std_rows)} standard class rows in report {i+1}")
        else:
            # Try alternate pattern - looser matching
            alt_rows = re.findall(r"\s*(\d+)(?:\s+|[:\.,])+(\d+\.\d+)(?:\s+|[:\.,])+(\d+\.\d+)(?:\s+|[:\.,])+(\d+\.\d+)(?:\s+|[:\.,])+(\d+)", report_chunk)
            class_rows = alt_rows
            if debug:
                print(f"Found {len(alt_rows)} alternate class rows in report {i+1}")
            
            # If still no results, try an even looser pattern
            if not alt_rows:
                # This pattern is very loose and might catch non-class rows, so use with caution
                super_loose = re.findall(r"(\d+)[\s:]*(\d+\.\d+)[\s:]*(\d+\.\d+)[\s:]*(\d+\.\d+)[\s:]*(\d+)", report_chunk)
                class_rows = super_loose
                if debug:
                    print(f"Found {len(super_loose)} super loose class rows in report {i+1}")
        
        # Try to match other report elements
        # Match accuracy
        accuracy_row = re.search(r"\s*accuracy\s+(\d+\.\d+)\s+(\d+)\s*", report_chunk)
        if not accuracy_row:
            # Try alternate pattern
            accuracy_row = re.search(r"accuracy.*?(\d+\.\d+).*?(\d+)", report_chunk)
        
        # Match macro avg
        macro_avg_row = re.search(r"\s*macro avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s*", report_chunk)
        if not macro_avg_row:
            # Try alternate pattern
            macro_avg_row = re.search(r"macro avg.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+)", report_chunk)
        
        # Match weighted avg
        weighted_avg_row = re.search(r"\s*weighted avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s*", report_chunk)
        if not weighted_avg_row:
            # Try alternate pattern
            weighted_avg_row = re.search(r"weighted avg.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+)", report_chunk)
        
        # Print what we found for debugging
        if debug:
            print(f"Report {i+1}: Found {len(class_rows)} class rows, " 
                  f"accuracy: {'Yes' if accuracy_row else 'No'}, "
                  f"macro avg: {'Yes' if macro_avg_row else 'No'}, "
                  f"weighted avg: {'Yes' if weighted_avg_row else 'No'}")
            
            if class_rows:
                print(f"Sample class row: {class_rows[0]}")
        
        # If we didn't find any class rows, try parsing the report as text
        if not class_rows and debug:
            print(f"No class rows found in report {i+1}, falling back to text parsing")
            # We'll proceed anyway to try the next fallback method
        
        # Extract context information (for stage/iteration)
        context_start = max(0, pos - 1000)
        context_before = log_content[context_start:pos]
        
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
        
        # Process class rows
        for class_row in class_rows:
            class_name, precision, recall, f1_score, support = class_row
            parsed_reports.append({
                'class': class_name,
                'precision': float(precision),
                'recall': float(recall),
                'f1-score': float(f1_score),
                'support': int(support),
                'iteration': iteration,
                'stage': stage,
                'report_index': i
            })
        
        # Process accuracy
        if accuracy_row:
            parsed_reports.append({
                'class': 'accuracy',
                'precision': None,
                'recall': None,
                'f1-score': float(accuracy_row.group(1)),
                'support': int(accuracy_row.group(2)),
                'iteration': iteration,
                'stage': stage,
                'report_index': i
            })
        
        # Process macro avg
        if macro_avg_row:
            parsed_reports.append({
                'class': 'macro avg',
                'precision': float(macro_avg_row.group(1)),
                'recall': float(macro_avg_row.group(2)),
                'f1-score': float(macro_avg_row.group(3)),
                'support': int(macro_avg_row.group(4)),
                'iteration': iteration,
                'stage': stage,
                'report_index': i
            })
        
        # Process weighted avg
        if weighted_avg_row:
            parsed_reports.append({
                'class': 'weighted avg',
                'precision': float(weighted_avg_row.group(1)),
                'recall': float(weighted_avg_row.group(2)),
                'f1-score': float(weighted_avg_row.group(3)),
                'support': int(weighted_avg_row.group(4)),
                'iteration': iteration,
                'stage': stage,
                'report_index': i
            })
    
    # If we found any reports, return them
    if parsed_reports:
        if debug:
            print(f"Successfully extracted {len(parsed_reports)} entries from classification reports")
        return parsed_reports
    
    # Fallback to using the full parse_classification_report approach
    if debug:
        print("\nNo reports found with pattern matching, trying direct text parsing...")
        
    # Try extracting report blocks and parsing them
    # This is now handled in the find_and_extract_reports function
    return []


def find_and_extract_reports(log_file_path, debug=False, identifier=None):
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
    
    # Try the new, more flexible approach first that can handle any class set
    parsed_reports = extract_reports_with_flexible_pattern(log_content, debug=debug)
    
    # If we got results, return them
    if parsed_reports:
        # Add identifier if provided
        if identifier:
            for report in parsed_reports:
                report['identifier'] = identifier
        return parsed_reports
    
    # If we still have no results, try a more general approach with parse_classification_report
    if debug:
        print("\nTrying a more general approach with parse_classification_report...")
    
    # Extract all classification reports as text blocks
    report_pattern = r"Classification report:[^\n]*\n((?:.*\n)+?)(?:\s*\n|\Z)"
    report_blocks = re.findall(report_pattern, log_content, re.DOTALL)
    
    if debug:
        print(f"Found {len(report_blocks)} classification report blocks")
        if report_blocks and debug:
            print(f"\nFirst report block sample:")
            print(report_blocks[0][:200])
    
    if not report_blocks:
        # Try another pattern that's more permissive
        report_pattern_alt = r"Classification report:.*?precision.*?recall.*?f1-score.*?support.*?((?:\s*\d+.*\n)+).*?accuracy.*\n.*?macro avg.*\n.*?weighted avg"
        report_blocks = re.findall(report_pattern_alt, log_content, re.DOTALL)
        if debug:
            print(f"Found {len(report_blocks)} classification report blocks with alternative pattern")
            if report_blocks:
                print(f"\nFirst alt report block sample:")
                print(report_blocks[0][:200])
                
        if not report_blocks:
            # Try an even more permissive pattern - just get chunks after "Classification report:"
            alt_pattern_2 = r"Classification report:(.*?)(?=Classification report:|$)"
            report_chunks = re.findall(alt_pattern_2, log_content, re.DOTALL)
            if debug:
                print(f"Found {len(report_chunks)} raw classification report chunks")
                
            # Process these chunks to find patterns within them
            for i, chunk in enumerate(report_chunks[:3]):  # Process first 3 chunks
                if debug:
                    print(f"\nRaw chunk {i+1} (first 200 chars):")
                    print(chunk[:200].replace('\n', ' '))
                    
                # Try to extract just the part with class numbers and metrics
                metrics_part = re.search(r"precision.*?recall.*?f1-score.*?support.*?((?:\s*\d+.*\n)+)", chunk, re.DOTALL)
                if metrics_part:
                    report_blocks.append(metrics_part.group(1))
                    if debug:
                        print(f"Extracted metrics part from chunk {i+1}")
    
    parsed_reports = []
    for i, report_block in enumerate(report_blocks):
        try:
            # Extract context information
            report_start_idx = log_content.find(report_block)
            if report_start_idx >= 0:
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
            
            # Use the parse_classification_report function
            report_data = parse_classification_report(report_block)
            
            if debug and i < 2 and report_data:
                print(f"Report {i+1}: Parsed {len(report_data)} metrics rows")
                if report_data:
                    print(f"First row: {report_data[0]}")
            
            # Add metadata to each row
            for row in report_data:
                row['iteration'] = iteration
                row['stage'] = stage
                row['report_index'] = i
                if identifier:
                    row['identifier'] = identifier
            
            parsed_reports.extend(report_data)
            
        except Exception as e:
            if debug:
                print(f"Error parsing report block {i}: {e}")
                print(f"Report content: {report_block[:200]}...")
            continue
    
    # If we still have no results, try direct pattern matching for class lines
    if not parsed_reports and debug:
        print("\nTrying direct pattern matching for class metrics lines...")
        
        # Try to find direct matches for class metrics lines
        class_pattern = r"\s*(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)"
        class_matches = re.findall(class_pattern, log_content)
        
        if debug:
            print(f"Found {len(class_matches)} direct class metric matches")
            if class_matches:
                print(f"First match: {class_matches[0]}")
        
        # If we found direct matches, convert them to reports
        if class_matches:
            for i, match in enumerate(class_matches):
                class_name, precision, recall, f1_score, support = match
                parsed_reports.append({
                    'class': class_name,
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1-score': float(f1_score),
                    'support': int(support),
                    'iteration': 'unknown',
                    'stage': 'unknown',
                    'report_index': i // 3  # Group every 3 classes as one report (heuristic)
                })
                if identifier:
                    parsed_reports[-1]['identifier'] = identifier
    
    return parsed_reports


def export_reports_to_csv(parsed_reports, output_dir, prefix="classification_reports", identifier=None):
    """Export the parsed reports to CSV files."""
    if not parsed_reports:
        print("No reports found to export.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(parsed_reports)
    
    # Construct filename prefix with identifier if provided
    file_prefix = prefix
    if identifier:
        file_prefix = f"{prefix}_{identifier}"
    
    # Save all reports to a single CSV
    all_reports_path = os.path.join(output_dir, "{0}_all.csv".format(file_prefix))
    df.to_csv(all_reports_path, index=False)
    print("Exported all classification reports to: {0}".format(all_reports_path))
    
    # Export separate CSVs for each iteration and stage
    # Group by iteration and stage
    for (iteration, stage), group_df in df.groupby(['iteration', 'stage']):
        # Skip groups with unknown iteration or stage if needed
        if iteration == "unknown" and stage == "unknown":
            continue
            
        # Create filename including identifier if provided
        if iteration != "unknown":
            filename = "{0}_iteration_{1}_stage_{2}.csv".format(file_prefix, iteration, stage)
        else:
            filename = "{0}_stage_{1}.csv".format(file_prefix, stage)
        
        # Export CSV
        csv_path = os.path.join(output_dir, filename)
        group_df.to_csv(csv_path, index=False)
        print("Exported report for iteration {0}, stage {1} to: {2}".format(
            iteration, stage, csv_path))
    
    # Summary creation has been removed as requested
    
    return all_reports_path


def main():
    parser = argparse.ArgumentParser(description='Extract classification reports from log files')
    parser.add_argument('--log_file', type=str, help='Path to the log file containing classification reports')
    parser.add_argument('--log_dir', type=str, help='Directory containing log files to process')
    parser.add_argument('--output_dir', type=str, default='classification_reports', 
                        help='Directory to save the extracted reports (default: classification_reports)')
    parser.add_argument('--prefix', type=str, default='classification_report',
                        help='Prefix for the CSV filenames (default: classification_report)')
    parser.add_argument('--identifier', type=str, 
                        help='Identifier to add as a column in CSV output and include in filenames')
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
        parsed_reports = find_and_extract_reports(args.log_file, debug=args.debug, identifier=args.identifier)
        all_parsed_reports.extend(parsed_reports)
    
    # Process all log files in directory
    if args.log_dir:
        if not os.path.exists(args.log_dir):
            parser.error("Log directory not found: {0}".format(args.log_dir))
        
        log_files = glob.glob(os.path.join(args.log_dir, "*.log")) + glob.glob(os.path.join(args.log_dir, "*.txt"))
        # Also look for files without extension (like HPC output files)
        for f in os.listdir(args.log_dir):
            path = os.path.join(args.log_dir, f)
            if os.path.isfile(path) and '.' not in f:
                log_files.append(path)
                
        if not log_files:
            parser.error("No log files found in directory: {0}".format(args.log_dir))
        
        for log_file in log_files:
            print("Processing log file: {0}".format(log_file))
            parsed_reports = find_and_extract_reports(log_file, debug=args.debug, identifier=args.identifier)
            if parsed_reports:
                all_parsed_reports.extend(parsed_reports)
                # Use the filename as identifier if none is provided
                if args.identifier is None:
                    file_identifier = os.path.basename(log_file)
                    for report in parsed_reports:
                        report['file_source'] = file_identifier
    
    # Export all parsed reports
    if all_parsed_reports:
        print("Found {0} report entries in total".format(len(all_parsed_reports)))
        export_reports_to_csv(all_parsed_reports, args.output_dir, args.prefix, args.identifier)
    else:
        print("No classification reports found in the provided log file(s)")


if __name__ == "__main__":
    main() 