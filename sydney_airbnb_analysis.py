#!/usr/bin/env python3
# Sydney Airbnb Data Analysis
# This script analyzes the Sydney Airbnb dataset to answer specific questions about hosts, properties, locations, and prices

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.2f}'.format)

print("Loading Sydney Airbnb dataset...")
# Load the dataset (using the correct filename)
df = pd.read_csv('Data/sydney_airbnb_data.csv')

print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# Display basic information about the dataset
print("\n=== DATASET OVERVIEW ===")
print(df.info())

# Display the first few rows
print("\n=== SAMPLE DATA ===")
print(df.head())

# Check for missing values
print("\n=== MISSING VALUES ===")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0].sort_values(ascending=False))

# ===== HOST ANALYSIS =====
print("\n\n===== HOST ANALYSIS =====")

# 1. Number of properties the avg/median Airbnb host has
print("\n=== NUMBER OF PROPERTIES PER HOST ===")
host_property_counts = df.groupby('host_id')['id'].count().reset_index()
host_property_counts.columns = ['host_id', 'property_count']

print(f"Average number of properties per host: {host_property_counts['property_count'].mean():.2f}")
print(f"Median number of properties per host: {host_property_counts['property_count'].median():.2f}")

# Distribution of property counts
print("\nDistribution of properties per host:")
property_count_distribution = host_property_counts['property_count'].value_counts().sort_index()
print(property_count_distribution)

# Hosts with the most properties
print("\nTop 10 hosts with the most properties:")
top_hosts = host_property_counts.sort_values('property_count', ascending=False).head(10)
top_hosts = top_hosts.merge(df[['host_id', 'host_name']].drop_duplicates(), on='host_id')
print(top_hosts[['host_name', 'property_count']])

# 2. Where are the Hosts from (host_location)
print("\n=== HOST LOCATIONS ===")
# Clean up host_location data
if 'host_location' in df.columns:
    # Extract country or city from host_location
    host_locations = df['host_location'].dropna().value_counts()
    
    print("Top 20 host locations:")
    print(host_locations.head(20))
    
    # Try to extract countries for a cleaner view
    def extract_country(location_str):
        if pd.isna(location_str):
            return np.nan
        
        # Common countries/regions in the data
        countries = ['Australia', 'USA', 'UK', 'New Zealand', 'Canada', 'Singapore', 
                    'Hong Kong', 'China', 'Japan', 'France', 'Germany', 'Italy']
        
        for country in countries:
            if country.lower() in location_str.lower():
                return country
        
        # Check for Australia specifically
        if 'sydney' in location_str.lower() or 'nsw' in location_str.lower():
            return 'Australia'
        
        # If no country found, return the original string
        return location_str
    
    df['host_country'] = df['host_location'].apply(extract_country)
    host_countries = df['host_country'].dropna().value_counts()
    
    print("\nHost countries (extracted):")
    print(host_countries.head(10))
    
    # Calculate percentage of local vs. international hosts
    local_hosts = df[df['host_country'] == 'Australia'].shape[0]
    total_hosts_with_location = df['host_country'].dropna().shape[0]
    
    print(f"\nPercentage of local (Australian) hosts: {local_hosts/total_hosts_with_location*100:.2f}%")
else:
    print("Host location data not available in the dataset")

# 3. Hosts period (how long have they been hosts)
print("\n=== HOST TENURE ===")
if 'host_since' in df.columns:
    # Convert host_since to datetime
    df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
    
    # Calculate hosting tenure in days and years
    current_date = pd.to_datetime('2025-07-05')  # Using the date from the metadata
    df['hosting_days'] = (current_date - df['host_since']).dt.days
    df['hosting_years'] = df['hosting_days'] / 365
    
    print(f"Average hosting tenure: {df['hosting_years'].mean():.2f} years")
    print(f"Median hosting tenure: {df['hosting_years'].median():.2f} years")
    
    # Distribution of hosting tenure
    print("\nHosting tenure distribution (years):")
    tenure_bins = [0, 1, 2, 3, 4, 5, 10, 15, 20]
    tenure_labels = ['<1', '1-2', '2-3', '3-4', '4-5', '5-10', '10-15', '15+']
    df['tenure_group'] = pd.cut(df['hosting_years'], bins=tenure_bins, labels=tenure_labels, right=False)
    tenure_distribution = df['tenure_group'].value_counts().sort_index()
    print(tenure_distribution)
    
    # Analyze if there's a correlation between tenure and number of properties
    host_tenure = df.groupby('host_id')['hosting_years'].first().reset_index()
    host_analysis = pd.merge(host_property_counts, host_tenure, on='host_id')
    
    correlation = host_analysis['property_count'].corr(host_analysis['hosting_years'])
    print(f"\nCorrelation between hosting tenure and number of properties: {correlation:.2f}")
else:
    print("Host since data not available in the dataset")

# ===== PROPERTY ANALYSIS =====
print("\n\n===== PROPERTY ANALYSIS =====")

# 1. Number of reviews
print("\n=== REVIEW STATISTICS ===")
if 'number_of_reviews' in df.columns:
    print(f"Average number of reviews per property: {df['number_of_reviews'].mean():.2f}")
    print(f"Median number of reviews per property: {df['number_of_reviews'].median():.2f}")
    print(f"Maximum number of reviews for a property: {df['number_of_reviews'].max()}")
    
    # Properties with most reviews
    print("\nTop 10 properties with most reviews:")
    top_reviewed = df.sort_values('number_of_reviews', ascending=False)[['name', 'number_of_reviews', 'host_name']].head(10)
    print(top_reviewed)
    
    # Distribution of review counts
    print("\nDistribution of review counts:")
    review_bins = [0, 1, 5, 10, 25, 50, 100, 500, 1000, df['number_of_reviews'].max() + 1]
    review_labels = ['0', '1-4', '5-9', '10-24', '25-49', '50-99', '100-499', '500-999', '1000+']
    df['review_count_group'] = pd.cut(df['number_of_reviews'], bins=review_bins, labels=review_labels, right=False)
    review_distribution = df['review_count_group'].value_counts().sort_index()
    print(review_distribution)
    
    # Correlation between reviews and other factors
    if 'review_scores_rating' in df.columns:
        print("\nCorrelation between number of reviews and rating:")
        rating_corr = df['number_of_reviews'].corr(df['review_scores_rating'])
        print(f"Correlation coefficient: {rating_corr:.2f}")
else:
    print("Review count data not available in the dataset")

# 2. Anything interesting within the reviews?
print("\n=== REVIEW ANALYSIS ===")
# Analyze review scores if available
review_score_columns = [col for col in df.columns if col.startswith('review_scores_')]
if review_score_columns:
    print("Average review scores by category:")
    for col in review_score_columns:
        print(f"{col.replace('review_scores_', '')}: {df[col].mean():.2f}")
    
    # Find properties with perfect scores
    perfect_scores = df[df['review_scores_rating'] == 5.0]
    print(f"\nNumber of properties with perfect 5.0 overall rating: {perfect_scores.shape[0]}")
    print(f"Percentage of perfect ratings: {perfect_scores.shape[0]/df.shape[0]*100:.2f}%")
    
    # Analyze if there's a correlation between price and ratings
    if 'price' in df.columns:
        # Extract numeric price
        df['price_numeric'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
        price_rating_corr = df['price_numeric'].corr(df['review_scores_rating'])
        print(f"\nCorrelation between price and rating: {price_rating_corr:.2f}")
else:
    print("Review score data not available in the dataset")

# ===== LOCATION ANALYSIS =====
print("\n\n===== LOCATION ANALYSIS =====")

# 1. Where are the Airbnbs generally located? Suburb/Neighbourhood
print("\n=== NEIGHBOURHOOD DISTRIBUTION ===")
if 'neighbourhood_cleansed' in df.columns:
    neighbourhood_counts = df['neighbourhood_cleansed'].value_counts()
    
    print("Top 20 neighbourhoods by listing count:")
    print(neighbourhood_counts.head(20))
    
    # Calculate percentage of listings in top neighbourhoods
    top_5_neighbourhoods = neighbourhood_counts.head(5).sum()
    total_listings = df.shape[0]
    print(f"\nPercentage of listings in top 5 neighbourhoods: {top_5_neighbourhoods/total_listings*100:.2f}%")
    
    # Average price by neighbourhood (if price data is available)
    if 'price' in df.columns:
        # Extract numeric price
        if 'price_numeric' not in df.columns:
            df['price_numeric'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
        
        neighbourhood_prices = df.groupby('neighbourhood_cleansed')['price_numeric'].agg(['mean', 'median', 'count']).sort_values('count', ascending=False)
        
        print("\nAverage and median prices in top 10 neighbourhoods:")
        print(neighbourhood_prices.head(10))
else:
    print("Neighbourhood data not available in the dataset")

# 2. Is there a lack of rentals within this area?
print("\n=== RENTAL AVAILABILITY ANALYSIS ===")
print("Note: To fully answer whether there's a lack of rentals in specific areas would require")
print("comparing with external data from REA Site. This analysis focuses on the availability")
print("metrics within the Airbnb dataset.")

# Analyze availability metrics if present
availability_cols = [col for col in df.columns if col.startswith('availability_')]
if availability_cols:
    print("\nAverage availability across different time periods:")
    for col in availability_cols:
        print(f"{col}: {df[col].mean():.2f} days")
    
    # Availability by neighbourhood
    if 'neighbourhood_cleansed' in df.columns and 'availability_365' in df.columns:
        neighbourhood_availability = df.groupby('neighbourhood_cleansed')['availability_365'].mean().sort_values()
        
        print("\nNeighbourhoods with lowest availability (365 days):")
        print(neighbourhood_availability.head(10))
        
        print("\nNeighbourhoods with highest availability (365 days):")
        print(neighbourhood_availability.tail(10))
else:
    print("Availability data not available in the dataset")

# ===== PRICE ANALYSIS =====
print("\n\n===== PRICE ANALYSIS =====")

# Average/median price per night
print("\n=== PRICE STATISTICS ===")
if 'price' in df.columns:
    # Ensure we have numeric price
    if 'price_numeric' not in df.columns:
        df['price_numeric'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
    
    print(f"Average price per night: ${df['price_numeric'].mean():.2f}")
    print(f"Median price per night: ${df['price_numeric'].median():.2f}")
    print(f"Minimum price: ${df['price_numeric'].min():.2f}")
    print(f"Maximum price: ${df['price_numeric'].max():.2f}")
    
    # Price distribution
    print("\nPrice distribution:")
    price_bins = [0, 50, 100, 150, 200, 300, 500, 1000, df['price_numeric'].max() + 1]
    price_labels = ['<$50', '$50-99', '$100-149', '$150-199', '$200-299', '$300-499', '$500-999', '$1000+']
    df['price_group'] = pd.cut(df['price_numeric'], bins=price_bins, labels=price_labels, right=False)
    price_distribution = df['price_group'].value_counts().sort_index()
    print(price_distribution)
    
    # Price by room type
    if 'room_type' in df.columns:
        room_type_prices = df.groupby('room_type')['price_numeric'].agg(['mean', 'median', 'count']).sort_values('count', ascending=False)
        print("\nPrices by room type:")
        print(room_type_prices)
    
    # Price by number of bedrooms
    if 'bedrooms' in df.columns:
        # Convert bedrooms to numeric if needed
        if df['bedrooms'].dtype == 'object':
            df['bedrooms_numeric'] = pd.to_numeric(df['bedrooms'], errors='coerce')
        else:
            df['bedrooms_numeric'] = df['bedrooms']
        
        bedroom_prices = df.groupby('bedrooms_numeric')['price_numeric'].agg(['mean', 'median', 'count']).sort_index()
        print("\nPrices by number of bedrooms:")
        print(bedroom_prices)
else:
    print("Price data not available in the dataset")

print("\n\nAnalysis complete!")
