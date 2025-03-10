import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import re

st.set_page_config(page_title="Relief Vehicle Dashboard", layout="wide", page_icon="🚌", initial_sidebar_state="collapsed")

# Set dark theme for the entire app
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    .stTitle, .stHeader, .stMarkdown {
        color: white !important;
    }
    .stSelectbox > div > div {
        background-color: #262730;
        color: white;
    }
    .stFileUploader {
        background-color: #262730;
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Relief Vehicle Overlap Dashboard")
st.markdown("Upload your relief vehicle data file to analyze vehicle overlaps at each stop.")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "xls", "csv"])

def convert_time_str_to_datetime(time_str, base_date=None):
    """Convert time string to datetime object"""
    if pd.isna(time_str):
        return None
    
    if base_date is None:
        base_date = datetime.today().date()
    
    if isinstance(time_str, str):
        try:
            # Handle AM/PM format like "10:49:00 AM"
            if "AM" in time_str or "PM" in time_str:
                # Try parsing with pandas first which handles various formats well
                try:
                    dt = pd.to_datetime(time_str).to_pydatetime()
                    return datetime.combine(base_date, dt.time())
                except:
                    # If pandas fails, try manual parsing
                    am_pm = "AM" if "AM" in time_str else "PM"
                    time_part = time_str.replace(am_pm, "").strip()
                    
                    time_parts = time_part.split(":")
                    hour = int(time_parts[0])
                    minute = int(time_parts[1]) if len(time_parts) > 1 else 0
                    second = int(time_parts[2]) if len(time_parts) > 2 else 0
                    
                    # Convert to 24-hour format if PM
                    if am_pm == "PM" and hour < 12:
                        hour += 12
                    if am_pm == "AM" and hour == 12:
                        hour = 0
                    
                    return datetime.combine(base_date, datetime.min.time().replace(hour=hour, minute=minute, second=second))
            
            # Handle standard time formats without AM/PM
            elif ":" in time_str:
                if len(time_str.split(':')) == 2:
                    hour, minute = map(int, time_str.split(':'))
                    # Handle 24-hour format correctly
                    days_to_add = hour // 24
                    hour = hour % 24
                    result = datetime.combine(base_date, datetime.min.time().replace(hour=hour, minute=minute))
                    if days_to_add > 0:
                        result += timedelta(days=days_to_add)
                    return result
                elif len(time_str.split(':')) == 3:
                    hour, minute, second = map(int, time_str.split(':'))
                    # Handle 24-hour format correctly
                    days_to_add = hour // 24
                    hour = hour % 24
                    result = datetime.combine(base_date, datetime.min.time().replace(hour=hour, minute=minute, second=second))
                    if days_to_add > 0:
                        result += timedelta(days=days_to_add)
                    return result
        except ValueError:
            # If parsing fails, try the most flexible approach with pandas
            try:
                dt = pd.to_datetime(time_str).to_pydatetime()
                return datetime.combine(base_date, dt.time())
            except:
                return None
    elif isinstance(time_str, datetime):
        return time_str
    
    return None

if uploaded_file is not None:
    try:
        # Read the file based on its type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, skiprows=5)
        
        st.success("File uploaded successfully!")
        
        # Display the raw data
        with st.expander("View Raw Data"):
            st.dataframe(df)
        
        # Ensure required columns exist
        required_columns = ["Relief Vehicle", "Origin", "Destination", "Start Time", "End Time"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            # Clean and prepare the data
            # Convert time strings to datetime
            base_date = datetime.today().date()
            df['Start Time'] = df['Start Time'].apply(lambda x: convert_time_str_to_datetime(x, base_date))
            df['End Time'] = df['End Time'].apply(lambda x: convert_time_str_to_datetime(x, base_date))
            
            # Get unique stops (from both Origin and Destination)
            all_stops = sorted(list(set(df['Origin'].unique()) | set(df['Destination'].unique())))
            
            # Create a dictionary to store vehicle presence at each stop
            stop_presence = {stop: [] for stop in all_stops}
            
            # Define buffer time in minutes (2.5 minutes as requested)
            BUFFER_MINUTES = 2.5
            
            # First, track all arrivals and departures for each vehicle
            arrivals = {}  # Dictionary to store arrivals by vehicle and stop
            departures = {}  # Dictionary to store departures by vehicle and stop
            
            # Process each relief vehicle separately
            for vehicle in df['Relief Vehicle'].unique():
                # Get trips for this vehicle and sort by start time
                vehicle_df = df[df['Relief Vehicle'] == vehicle].sort_values('Start Time')
                
                # Analyze each trip for this vehicle
                for i, current_trip in vehicle_df.iterrows():
                    # Find the next trip for this vehicle (if any)
                    next_trips = vehicle_df[vehicle_df['Start Time'] > current_trip['End Time']].sort_values('Start Time')
                    next_trip = next_trips.iloc[0] if not next_trips.empty else None
                    
                    # ---- Track departure from origin ----
                    if not pd.isna(current_trip['Origin']) and not pd.isna(current_trip['Start Time']):
                        origin = current_trip['Origin']
                        start_time = current_trip['Start Time']
                        
                        # Create key for vehicle+stop
                        key = (vehicle, origin)
                        
                        # Store departure info
                        if key not in departures:
                            departures[key] = []
                            
                        departures[key].append({
                            'time': start_time,
                            'trip_id': i,
                            'direction': current_trip.get('Relief Vehicle Direction', 'Unknown'),
                            'prev_block': current_trip.get('Previous Block', ''),
                            'prev_route': current_trip.get('Previous Route', ''),
                            'next_block': current_trip.get('Next Block', ''),
                            'next_route': current_trip.get('Next Route', '')
                        })
                    
                    # ---- Track arrival at destination ----
                    if not pd.isna(current_trip['Destination']) and not pd.isna(current_trip['End Time']):
                        destination = current_trip['Destination']
                        end_time = current_trip['End Time']
                        
                        # Create key for vehicle+stop
                        key = (vehicle, destination)
                        
                        # Store arrival info
                        if key not in arrivals:
                            arrivals[key] = []
                            
                        arrivals[key].append({
                            'time': end_time,
                            'trip_id': i,
                            'direction': current_trip.get('Relief Vehicle Direction', 'Unknown'),
                            'prev_block': current_trip.get('Previous Block', ''),
                            'prev_route': current_trip.get('Previous Route', ''),
                            'next_block': current_trip.get('Next Block', ''),
                            'next_route': current_trip.get('Next Route', '')
                        })
            
            # Now process the arrivals and departures to create stop presence records
            for key in set(list(arrivals.keys()) + list(departures.keys())):
                vehicle, stop = key
                
                vehicle_arrivals = sorted(arrivals.get(key, []), key=lambda x: x['time'])
                vehicle_departures = sorted(departures.get(key, []), key=lambda x: x['time'])
                
                # Case 1: Vehicle arrives at stop and then departs later (extended stay)
                if vehicle_arrivals and vehicle_departures:
                    # Match arrivals with subsequent departures to create extended stays
                    # We'll use a greedy approach - match each arrival with the next available departure
                    
                    # Create a copy of departures we can modify
                    available_departures = vehicle_departures.copy()
                    
                    for arrival in vehicle_arrivals:
                        arrival_time = arrival['time']
                        arrival_trip = arrival['trip_id']
                        
                        # Find the next departure after this arrival
                        matching_departure = None
                        matching_idx = None
                        
                        for idx, departure in enumerate(available_departures):
                            if departure['time'] >= arrival_time:
                                # Only match if the departure is part of a different trip
                                # This prevents matching an arrival with its own departure at the same time
                                if departure['trip_id'] != arrival_trip or departure['time'] > arrival_time:
                                    matching_departure = departure
                                    matching_idx = idx
                                    break
                        
                        if matching_departure:
                            # Found a matching departure - remove it from available list
                            del available_departures[matching_idx]
                            
                            # Get a unique ID for this stay
                            stay_id = f"{arrival['trip_id']}-{matching_departure['trip_id']}"
                            
                            # Create a record for the stay with buffer
                            stay_start = arrival_time - timedelta(minutes=BUFFER_MINUTES)
                            stay_end = matching_departure['time'] + timedelta(minutes=BUFFER_MINUTES)
                            
                            # Check if this is an instant stay (arrival and departure at same time or very close)
                            is_instant = (matching_departure['time'] - arrival_time).total_seconds() < 60
                            
                            # Add the stay record
                            stop_presence[stop].append({
                                'vehicle': vehicle,
                                'start': stay_start,
                                'end': stay_end,
                                'direction': arrival['direction'],
                                'event_type': 'Instant Stop' if is_instant else 'Extended Stay',
                                'stay_id': stay_id,  # Unique ID for this stay
                                'actual_arrival': arrival_time,
                                'actual_departure': matching_departure['time'],
                                'arrival_trip': arrival['trip_id'],
                                'departure_trip': matching_departure['trip_id'],
                                'prev_block': arrival['prev_block'],
                                'prev_route': arrival['prev_route'],
                                'next_block': matching_departure['next_block'],
                                'next_route': matching_departure['next_route']
                            })
                        else:
                            # No matching departure found, just record the arrival with buffer
                            stop_presence[stop].append({
                                'vehicle': vehicle,
                                'start': arrival_time - timedelta(minutes=BUFFER_MINUTES),
                                'end': arrival_time + timedelta(minutes=BUFFER_MINUTES),
                                'direction': arrival['direction'],
                                'event_type': 'Arrival Only',
                                'stay_id': f"{arrival['trip_id']}-end",
                                'actual_arrival': arrival_time,
                                'actual_departure': None
                            })
                    
                    # Handle any remaining departures that had no matching arrival
                    for departure in available_departures:
                        departure_time = departure['time']
                        
                        stop_presence[stop].append({
                            'vehicle': vehicle,
                            'start': departure_time - timedelta(minutes=BUFFER_MINUTES),
                            'end': departure_time + timedelta(minutes=BUFFER_MINUTES),
                            'direction': departure['direction'],
                            'event_type': 'Departure Only',
                            'stay_id': f"start-{departure['trip_id']}",
                            'actual_arrival': None,
                            'actual_departure': departure_time
                        })
                
                # Case 2: Only arrivals, no departures
                elif vehicle_arrivals:
                    for arrival in vehicle_arrivals:
                        arrival_time = arrival['time']
                        
                        stop_presence[stop].append({
                            'vehicle': vehicle,
                            'start': arrival_time - timedelta(minutes=BUFFER_MINUTES),
                            'end': arrival_time + timedelta(minutes=BUFFER_MINUTES),
                            'direction': arrival['direction'],
                            'event_type': 'Arrival Only',
                            'stay_id': f"{arrival['trip_id']}-end",
                            'actual_arrival': arrival_time,
                            'actual_departure': None
                        })
                
                # Case 3: Only departures, no arrivals
                elif vehicle_departures:
                    for departure in vehicle_departures:
                        departure_time = departure['time']
                        
                        stop_presence[stop].append({
                            'vehicle': vehicle,
                            'start': departure_time - timedelta(minutes=BUFFER_MINUTES),
                            'end': departure_time + timedelta(minutes=BUFFER_MINUTES),
                            'direction': departure['direction'],
                            'event_type': 'Departure Only',
                            'stay_id': f"start-{departure['trip_id']}",
                            'actual_arrival': None,
                            'actual_departure': departure_time
                        })
            
            # Create dropdown for stops instead of tabs
            if all_stops:
                # Prepare colors for different vehicles
                unique_vehicles = sorted(df['Relief Vehicle'].unique())
                colorscale = [f"hsl({i*360/len(unique_vehicles)}, 70%, 50%)" for i in range(len(unique_vehicles))]
                vehicle_colors = {vehicle: colorscale[i % len(colorscale)] for i, vehicle in enumerate(unique_vehicles)}
                
                # Create a dropdown for stop selection
                selected_stop = st.selectbox("Select a stop to display", all_stops)
                
                # Process the selected stop
                stop = selected_stop
                st.header(f"Relief Vehicles at {stop}")
                
                if not stop_presence[stop]:
                    st.info(f"No relief vehicles recorded at {stop}")
                else:
                    # Sort the presence data by start time
                    stop_presence[stop].sort(key=lambda x: x['start'])
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Merge overlapping time blocks for the same vehicle
                    # First, sort by vehicle and start time
                    timeline_data = sorted(stop_presence[stop], key=lambda x: (x['vehicle'], x['start']))
                    
                    # We'll be more careful about merging - only merge entries with the same stay_id
                    merged_timeline = []
                    current_block = None
                    
                    for entry in timeline_data:
                        if entry['start'] is None or entry['end'] is None:
                            continue
                            
                        if current_block is None:
                            current_block = entry.copy()
                        elif (current_block['vehicle'] == entry['vehicle'] and
                              entry['start'] <= current_block['end'] and 
                              current_block.get('stay_id') == entry.get('stay_id')):
                            # Blocks overlap AND belong to the same stay - extend current block
                            current_block['end'] = max(current_block['end'], entry['end'])
                        else:
                            # Different stay or no overlap - add the current block and start a new one
                            merged_timeline.append(current_block)
                            current_block = entry.copy()
                    
                    # Add the last block if any
                    if current_block is not None:
                        merged_timeline.append(current_block)
                    
                    # Use the merged timeline data for the chart
                    timeline_data = merged_timeline
                    
                    # Debug output for troubleshooting (hidden in collapsed section)
                    with st.expander("Debug Timeline Data"):
                        debug_data = []
                        for entry in timeline_data:
                            debug_entry = {
                                'Vehicle': entry['vehicle'],
                                'Start': entry['start'].strftime("%H:%M:%S"),
                                'End': entry['end'].strftime("%H:%M:%S"),
                                'Duration (min)': round((entry['end'] - entry['start']).total_seconds() / 60, 1),
                                'Type': entry.get('event_type', 'Unknown'),
                                'Stay ID': entry.get('stay_id', 'Unknown')
                            }
                            
                            # Add actual arrival/departure times if available
                            if 'actual_arrival' in entry and entry['actual_arrival'] is not None:
                                debug_entry['Actual Arrival'] = entry['actual_arrival'].strftime("%H:%M:%S")
                            if 'actual_departure' in entry and entry['actual_departure'] is not None:
                                debug_entry['Actual Departure'] = entry['actual_departure'].strftime("%H:%M:%S")
                                
                            debug_data.append(debug_entry)
                        st.dataframe(debug_data)
                    
                    # Sort timeline data by start time
                    timeline_data.sort(key=lambda x: x['start'])
                    
                    # Auto-zoom: Find the first block's time to center the view
                    if timeline_data:
                        first_block_time = timeline_data[0]['start']
                        zoom_start = first_block_time
                        zoom_end = first_block_time + timedelta(hours=2)  # Show 2-hour window
                    else:
                        # Default if no data
                        zoom_start = datetime.combine(base_date, datetime.min.time()).replace(hour=8)
                        zoom_end = datetime.combine(base_date, datetime.min.time()).replace(hour=10)
                    
                    # Simple approach: assign rows purely based on time overlap
                    # This guarantees that a vehicle will always be placed in the lowest free row
                    
                    # Sort data by start time for consistent processing
                    timeline_data.sort(key=lambda x: x['start'])
                    
                    # Process each vehicle and assign a row
                    for entry in timeline_data:
                        # Try each row starting from 0 until we find one with no conflict
                        row = 0
                        while True:
                            conflict = False
                            
                            # Check for conflicts in this row
                            for other in timeline_data:
                                if other == entry:  # Skip self
                                    continue
                                    
                                # Skip entries that don't have a position yet
                                if 'y_pos' not in other:
                                    continue
                                    
                                # Check if other entry is in this row and times overlap
                                if other['y_pos'] == row:
                                    # Format the start/end times for logging
                                    entry_start = entry['start'].strftime("%H:%M")
                                    entry_end = entry['end'].strftime("%H:%M")
                                    other_start = other['start'].strftime("%H:%M")
                                    other_end = other['end'].strftime("%H:%M")
                                    
                                    # Check for time overlap - the times should not overlap at all
                                    # Two vehicles overlap if one starts before the other ends and ends after the other starts
                                    # Exact matches (end time = start time) don't count as overlaps
                                    if (entry['start'] < other['end'] and entry['end'] > other['start']):
                                        conflict = True
                                        break
                            
                            if not conflict:
                                # This row works, use it
                                break
                                
                            # Try next row
                            row += 1
                            
                        # Assign this row
                        entry['y_pos'] = row
                    
                    # Dictionary to keep track of vehicle durations for the legend
                    vehicle_durations = {}
                    added_vehicles = set()
                    
                    # Add rectangles for vehicle presence
                    for entry in timeline_data:
                        vehicle = entry['vehicle']
                        start_time = entry['start']
                        end_time = entry['end']
                        y_pos = entry.get('y_pos', 0)
                        duration = (end_time - start_time).total_seconds() / 60
                        
                        # Format the start and end times as HH:MM
                        start_time_str = start_time.strftime("%H:%M")
                        end_time_str = end_time.strftime("%H:%M")
                        
                        # Get more descriptive information for the hover text
                        event_type = entry.get('event_type', 'Unknown')
                        
                        # Prepare the hover text with detailed information
                        hover_text = f"{vehicle} ({event_type})<br>"
                        hover_text += f"Display Time: {start_time_str} to {end_time_str} ({int(duration)} min)<br>"
                        
                        if 'actual_arrival' in entry and entry['actual_arrival'] is not None:
                            hover_text += f"Actual Arrival: {entry['actual_arrival'].strftime('%H:%M:%S')}<br>"
                        if 'actual_departure' in entry and entry['actual_departure'] is not None:
                            hover_text += f"Actual Departure: {entry['actual_departure'].strftime('%H:%M:%S')}<br>"
                            
                        # Add route information if available
                        if 'prev_route' in entry and entry['prev_route']:
                            hover_text += f"Previous Route: {entry['prev_route']}<br>"
                        if 'next_route' in entry and entry['next_route']:
                            hover_text += f"Next Route: {entry['next_route']}<br>"
                        
                        # Add rectangle with hover information
                        fig.add_trace(go.Scatter(
                            x=[start_time, end_time, end_time, start_time, start_time],
                            y=[y_pos, y_pos, y_pos+1, y_pos+1, y_pos],
                            fill="toself",
                            fillcolor=vehicle_colors[vehicle],
                            line=dict(color=vehicle_colors[vehicle], width=1),
                            mode="lines",
                            name=f"{vehicle}: {int(duration)} min ({start_time_str} to {end_time_str})",
                            hoverinfo="text",
                            hovertext=hover_text,
                            showlegend=(vehicle not in added_vehicles)
                        ))
                        
                        if vehicle not in added_vehicles:
                            added_vehicles.add(vehicle)
                    
                    # Sort the legend traces by vehicle number
                    # First extract all trace indices that have a legend entry
                    legend_traces = []
                    for i, trace in enumerate(fig.data):
                        if trace.showlegend:
                            # Extract vehicle number from name using regex
                            match = re.search(r'^(.*?):', trace.name)
                            if match:
                                vehicle_name = match.group(1)
                                # Try to extract a numeric part for sorting
                                numeric_part = ''.join(filter(str.isdigit, vehicle_name))
                                if numeric_part:
                                    sort_key = int(numeric_part)
                                else:
                                    sort_key = vehicle_name  # Non-numeric name
                                legend_traces.append((i, sort_key, trace.name))
                    
                    # Sort the legend entries by vehicle number
                    legend_traces.sort(key=lambda x: x[1])
                    
                    # Apply the sorted order
                    for i, (trace_idx, _, _) in enumerate(legend_traces):
                        fig.data[trace_idx].legendgroup = str(i)
                        fig.data[trace_idx].legendrank = i
                    
                    # Configure the layout
                    fig.update_layout(
                        title=f"Relief Vehicles at {stop}",
                        xaxis=dict(
                            title="Time of Day",
                            type="date",
                            tickformat="%H:%M",
                            range=[
                                datetime.combine(base_date, datetime.min.time()).replace(hour=6),  # Start at 6am
                                datetime.combine(base_date, datetime.min.time()).replace(hour=22)  # End at 10pm
                            ],
                            gridwidth=1,
                            gridcolor='rgba(255,255,255,0.1)',
                            color='white'  # White text for time labels
                        ),
                        yaxis=dict(
                            title="Vehicles",
                            showticklabels=False,
                            autorange=True,  # Enable auto-ranging
                            color='white'  # White text for axis title
                        ),
                        height=600,  # Fixed height for the graph
                        legend=dict(
                            orientation="h", 
                            yanchor="bottom", 
                            y=1.02, 
                            xanchor="right", 
                            x=1,
                            font=dict(color='white'),  # White text for legend
                            traceorder="grouped"  # Use the legendrank for ordering
                        ),
                        margin=dict(l=50, r=50, b=50, t=80, pad=4),
                        plot_bgcolor='black',  # Black background
                        paper_bgcolor='black',  # Black paper background
                        font=dict(color='white')  # White text throughout
                    )
                    
                    # Plot the figure
                    st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        st.code(traceback.format_exc())
else:
    st.info("Please upload a file to begin analysis.")
