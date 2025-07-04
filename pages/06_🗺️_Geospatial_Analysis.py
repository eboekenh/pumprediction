import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
import geopandas as gpd
from src.logger import logging
from src.exception import CustomException
import os
import sys

st.set_page_config(page_title="Geospatial Analysis", page_icon="🗺️", layout="wide")

def load_data():
    """Load data from session state or file"""
    try:
        if 'data_path' in st.session_state and st.session_state.data_path:
            df = pd.read_csv(st.session_state.data_path)
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def filter_valid_coordinates(df):
    """Filter data to include only valid coordinates"""
    if 'latitude' in df.columns and 'longitude' in df.columns:
        valid_data = df[
            (df['latitude'] != 0) & 
            (df['longitude'] != 0) & 
            (df['latitude'].notna()) & 
            (df['longitude'].notna()) &
            (df['latitude'].between(-90, 90)) &
            (df['longitude'].between(-180, 180))
        ].copy()
        return valid_data
    else:
        return pd.DataFrame()

def display_map_overview(df):
    """Display overview map of water pumps"""
    st.subheader("🌍 Water Pump Distribution Map")
    
    valid_data = filter_valid_coordinates(df)
    
    if len(valid_data) == 0:
        st.warning("No valid coordinates found in the dataset.")
        return
    
    # Sample data for performance if dataset is large
    sample_size = min(2000, len(valid_data))
    if len(valid_data) > sample_size:
        sample_data = valid_data.sample(n=sample_size, random_state=42)
        st.info(f"Showing {sample_size} randomly sampled points out of {len(valid_data)} total points for better performance.")
    else:
        sample_data = valid_data
    
    # Map configuration
    map_style = st.selectbox(
        "Select map style:",
        ["OpenStreetMap", "Satellite", "Terrain"],
        index=0
    )
    
    # Create base map
    center_lat = sample_data['latitude'].mean()
    center_lon = sample_data['longitude'].mean()
    
    tiles_dict = {
        "OpenStreetMap": "OpenStreetMap",
        "Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Terrain": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}"
    }
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles=tiles_dict[map_style],
        attr="Map data"
    )
    
    # Add markers based on status
    if 'status_group' in sample_data.columns:
        color_map = {
            'functional': 'green',
            'functional needs repair': 'orange',
            'non functional': 'red'
        }
        
        # Create marker clusters for better performance
        marker_cluster = MarkerCluster().add_to(m)
        
        for idx, row in sample_data.iterrows():
            status = row['status_group']
            color = color_map.get(status, 'blue')
            
            popup_text = f"""
            <b>Status:</b> {status}<br>
            <b>Region:</b> {row.get('region', 'Unknown')}<br>
            <b>District:</b> {row.get('lga', 'Unknown')}<br>
            <b>Water Quality:</b> {row.get('water_quality', 'Unknown')}<br>
            <b>Quantity:</b> {row.get('quantity', 'Unknown')}<br>
            <b>Management:</b> {row.get('management', 'Unknown')}<br>
            <b>Construction Year:</b> {row.get('construction_year', 'Unknown')}
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"Status: {status}"
            ).add_to(marker_cluster)
        
        # Add legend
        legend_html = """
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>Water Pump Status</h4>
        <p><i class="fa fa-circle" style="color:green"></i> Functional</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Needs Repair</p>
        <p><i class="fa fa-circle" style="color:red"></i> Non-functional</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
    
    else:
        # No status information available
        for idx, row in sample_data.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color='blue',
                fill=True,
                popup=f"Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f}"
            ).add_to(m)
    
    # Display map
    st_folium(m, width=700, height=500)

def display_heatmap(df):
    """Display heatmap of water pump density"""
    st.subheader("🔥 Water Pump Density Heatmap")
    
    valid_data = filter_valid_coordinates(df)
    
    if len(valid_data) == 0:
        st.warning("No valid coordinates found for heatmap.")
        return
    
    # Sample data for performance
    sample_size = min(1000, len(valid_data))
    sample_data = valid_data.sample(n=sample_size, random_state=42)
    
    # Create heatmap
    center_lat = sample_data['latitude'].mean()
    center_lon = sample_data['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Prepare data for heatmap
    heat_data = [[row['latitude'], row['longitude']] for idx, row in sample_data.iterrows()]
    
    # Add heatmap layer
    HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
    
    # Display map
    st_folium(m, width=700, height=500)

def display_regional_analysis(df):
    """Display regional analysis of water pumps"""
    st.subheader("📊 Regional Analysis")
    
    if 'region' not in df.columns:
        st.warning("Region information not available in the dataset.")
        return
    
    # Regional distribution
    regional_counts = df['region'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart of regional distribution
        fig_bar = px.bar(
            x=regional_counts.index,
            y=regional_counts.values,
            title='Water Pumps by Region',
            labels={'x': 'Region', 'y': 'Number of Pumps'}
        )
        fig_bar.update_xaxes(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Pie chart of regional distribution
        fig_pie = px.pie(
            values=regional_counts.values,
            names=regional_counts.index,
            title='Regional Distribution (%)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Status by region if available
    if 'status_group' in df.columns:
        st.subheader("🎯 Status Distribution by Region")
        
        # Cross-tabulation
        status_by_region = pd.crosstab(df['region'], df['status_group'])
        
        # Stacked bar chart
        fig_stack = px.bar(
            status_by_region.reset_index(),
            x='region',
            y=['functional', 'functional needs repair', 'non functional'],
            title='Water Pump Status by Region',
            barmode='stack'
        )
        fig_stack.update_xaxes(tickangle=45)
        st.plotly_chart(fig_stack, use_container_width=True)
        
        # Percentage breakdown
        status_by_region_pct = status_by_region.div(status_by_region.sum(axis=1), axis=0) * 100
        
        fig_pct = px.bar(
            status_by_region_pct.reset_index(),
            x='region',
            y=['functional', 'functional needs repair', 'non functional'],
            title='Water Pump Status by Region (%)',
            barmode='stack'
        )
        fig_pct.update_xaxes(tickangle=45)
        st.plotly_chart(fig_pct, use_container_width=True)

def display_water_quality_analysis(df):
    """Display water quality geographic analysis"""
    st.subheader("💧 Water Quality Geographic Analysis")
    
    if 'water_quality' not in df.columns:
        st.warning("Water quality information not available in the dataset.")
        return
    
    valid_data = filter_valid_coordinates(df)
    
    if len(valid_data) == 0:
        st.warning("No valid coordinates found for water quality analysis.")
        return
    
    # Water quality distribution
    quality_counts = df['water_quality'].value_counts()
    
    # Display distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_quality = px.bar(
            x=quality_counts.index,
            y=quality_counts.values,
            title='Water Quality Distribution',
            labels={'x': 'Water Quality', 'y': 'Count'}
        )
        st.plotly_chart(fig_quality, use_container_width=True)
    
    with col2:
        # Pie chart
        fig_pie = px.pie(
            values=quality_counts.values,
            names=quality_counts.index,
            title='Water Quality Distribution (%)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Geographic distribution of water quality
    if len(valid_data) > 0:
        st.subheader("🗺️ Water Quality Geographic Distribution")
        
        # Sample data for performance
        sample_size = min(1000, len(valid_data))
        sample_data = valid_data.sample(n=sample_size, random_state=42)
        
        # Create map
        center_lat = sample_data['latitude'].mean()
        center_lon = sample_data['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Color mapping for water quality
        quality_colors = {
            'soft': 'blue',
            'salty': 'red',
            'milky': 'white',
            'colored': 'purple',
            'fluoride': 'yellow',
            'unknown': 'gray'
        }
        
        # Add markers
        for idx, row in sample_data.iterrows():
            quality = row['water_quality']
            color = quality_colors.get(quality, 'gray')
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=f"Water Quality: {quality}",
                tooltip=f"Quality: {quality}"
            ).add_to(m)
        
        # Display map
        st_folium(m, width=700, height=500)

def display_infrastructure_analysis(df):
    """Display infrastructure analysis"""
    st.subheader("🏗️ Infrastructure Analysis")
    
    # Pump type analysis
    if 'pump_type' in df.columns:
        st.write("**Pump Type Distribution**")
        pump_counts = df['pump_type'].value_counts().head(10)
        
        fig_pump = px.bar(
            x=pump_counts.index,
            y=pump_counts.values,
            title='Top 10 Pump Types',
            labels={'x': 'Pump Type', 'y': 'Count'}
        )
        fig_pump.update_xaxes(tickangle=45)
        st.plotly_chart(fig_pump, use_container_width=True)
    
    # Construction year analysis
    if 'construction_year' in df.columns:
        st.write("**Construction Year Analysis**")
        
        # Filter valid years
        valid_years = df[(df['construction_year'] > 1900) & (df['construction_year'] <= 2024)]
        
        if len(valid_years) > 0:
            fig_year = px.histogram(
                valid_years,
                x='construction_year',
                nbins=20,
                title='Water Pump Construction Timeline'
            )
            st.plotly_chart(fig_year, use_container_width=True)
            
            # Age analysis
            current_year = 2024
            valid_years['pump_age'] = current_year - valid_years['construction_year']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Pump Age", f"{valid_years['pump_age'].mean():.1f} years")
            
            with col2:
                st.metric("Oldest Pump", f"{valid_years['pump_age'].max():.0f} years")
    
    # Management analysis
    if 'management' in df.columns:
        st.write("**Management Distribution**")
        mgmt_counts = df['management'].value_counts().head(10)
        
        fig_mgmt = px.pie(
            values=mgmt_counts.values,
            names=mgmt_counts.index,
            title='Water Pump Management Types'
        )
        st.plotly_chart(fig_mgmt, use_container_width=True)

def display_predictive_geospatial_analysis(df):
    """Display predictive geospatial analysis if predictions are available"""
    st.subheader("🔮 Predictive Geospatial Analysis")
    
    # Check if predictions are available
    if 'predicted_status' in df.columns:
        st.write("Displaying predictions on map...")
        
        valid_data = filter_valid_coordinates(df)
        
        if len(valid_data) == 0:
            st.warning("No valid coordinates found for prediction mapping.")
            return
        
        # Sample data for performance
        sample_size = min(1000, len(valid_data))
        sample_data = valid_data.sample(n=sample_size, random_state=42)
        
        # Create map
        center_lat = sample_data['latitude'].mean()
        center_lon = sample_data['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Color mapping for predictions
        pred_colors = {
            'functional': 'green',
            'functional needs repair': 'orange',
            'non functional': 'red'
        }
        
        # Add markers
        for idx, row in sample_data.iterrows():
            prediction = row['predicted_status']
            color = pred_colors.get(prediction, 'blue')
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=f"Predicted: {prediction}",
                tooltip=f"Prediction: {prediction}"
            ).add_to(m)
        
        # Display map
        st_folium(m, width=700, height=500)
        
        # Prediction statistics
        pred_counts = df['predicted_status'].value_counts()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Functional", pred_counts.get('functional', 0))
        
        with col2:
            st.metric("Predicted Needs Repair", pred_counts.get('functional needs repair', 0))
        
        with col3:
            st.metric("Predicted Non-functional", pred_counts.get('non functional', 0))
    
    else:
        st.info("No prediction data available. Run predictions first to see predictive analysis.")

def main():
    """Main geospatial analysis page function"""
    st.title("🗺️ Geospatial Analysis")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Display dataset info
        st.subheader("📊 Dataset Information")
        
        valid_coords = filter_valid_coordinates(df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            st.metric("Valid Coordinates", f"{len(valid_coords):,}")
        
        with col3:
            coord_percentage = (len(valid_coords) / len(df)) * 100 if len(df) > 0 else 0
            st.metric("Coordinate Coverage", f"{coord_percentage:.1f}%")
        
        with col4:
            regions = df['region'].nunique() if 'region' in df.columns else 0
            st.metric("Regions", regions)
        
        # Analysis tabs
        tabs = st.tabs([
            "Map Overview",
            "Density Heatmap",
            "Regional Analysis",
            "Water Quality",
            "Infrastructure",
            "Predictive Analysis"
        ])
        
        with tabs[0]:
            display_map_overview(df)
        
        with tabs[1]:
            display_heatmap(df)
        
        with tabs[2]:
            display_regional_analysis(df)
        
        with tabs[3]:
            display_water_quality_analysis(df)
        
        with tabs[4]:
            display_infrastructure_analysis(df)
        
        with tabs[5]:
            display_predictive_geospatial_analysis(df)
        
        # Export options
        st.subheader("📥 Export Options")
        
        if st.button("📊 Generate Geospatial Report"):
            try:
                # Create summary report
                report = f"""
# Geospatial Analysis Report

## Dataset Overview
- Total Records: {len(df):,}
- Valid Coordinates: {len(valid_coords):,}
- Coordinate Coverage: {coord_percentage:.1f}%

## Regional Distribution
{df['region'].value_counts().to_string() if 'region' in df.columns else 'Region data not available'}

## Water Quality Distribution
{df['water_quality'].value_counts().to_string() if 'water_quality' in df.columns else 'Water quality data not available'}

## Infrastructure Summary
- Pump Types: {df['pump_type'].nunique() if 'pump_type' in df.columns else 'N/A'}
- Management Types: {df['management'].nunique() if 'management' in df.columns else 'N/A'}
- Construction Years: {df['construction_year'].min() if 'construction_year' in df.columns else 'N/A'} - {df['construction_year'].max() if 'construction_year' in df.columns else 'N/A'}
                """
                
                st.download_button(
                    label="Download Geospatial Report",
                    data=report,
                    file_name="geospatial_analysis_report.txt",
                    mime="text/plain"
                )
            
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    
    else:
        st.warning("⚠️ No data loaded. Please upload data from the main page first.")
        
        if st.button("🏠 Go to Main Page"):
            st.switch_page("app.py")

if __name__ == "__main__":
    main()
