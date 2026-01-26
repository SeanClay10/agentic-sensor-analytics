"""
SMT Analytics API client for accessing sensor data.
Implements the SMT Analytics API specification v1.1.
"""

import requests
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict
from datetime import datetime
import time

from .models import SensorMetadata, NodeMetadata, SensorReading
from .config import DataConfig


class SMTAPIError(Exception):
    """Base exception for SMT API errors."""
    pass


class SMTAuthenticationError(SMTAPIError):
    """Raised when authentication fails."""
    pass


class SMTAPIClient:
    """
    Client for SMT Analytics API.
    Handles authentication, session management, and data retrieval.
    """
    
    def __init__(
        self,
        username: str,
        password: str,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize SMT API client.
        
        Args:
            username: API username
            password: API password
            base_url: Base URL for API endpoints
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
        """
        self.username = username
        self.password = password
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.session = requests.Session()
        self.session_id: Optional[str] = None
        self.authenticated = False
    
    @classmethod
    def from_config(cls, config: DataConfig) -> 'SMTAPIClient':
        """
        Create API client from configuration.
        
        Args:
            config: DataConfig object
            
        Returns:
            Configured SMTAPIClient instance
        """
        return cls(
            username=config.api.username,
            password=config.api.password,
            base_url=config.api.base_url,
            timeout=config.api.timeout,
            max_retries=config.api.max_retries
        )
    
    def _make_request(
        self,
        action: str,
        params: Optional[Dict[str, str]] = None
    ) -> ET.Element:
        """
        Make API request with retry logic.
        
        Args:
            action: API action to perform
            params: Optional query parameters
            
        Returns:
            Parsed XML root element
            
        Raises:
            SMTAPIError: If request fails after retries
        """
        if params is None:
            params = {}
        
        params['action'] = action
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.content)
                
                # Check for error in response
                error = root.find('error')
                if error is not None:
                    error_msg = error.text or "Unknown error"
                    raise SMTAPIError(f"API error: {error_msg}")
                
                return root
                
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    raise SMTAPIError(f"Request failed after {self.max_retries} attempts: {e}")
        
        raise SMTAPIError("Unexpected error in request")
    
    def login(self) -> bool:
        """
        Authenticate with the API and establish session.
        
        Returns:
            True if authentication successful
            
        Raises:
            SMTAuthenticationError: If authentication fails
        """
        try:
            root = self._make_request(
                action='login',
                params={
                    'user_username': self.username,
                    'user_password': self.password
                }
            )
            
            # Check for success
            login_result = root.find('login')
            if login_result is not None and login_result.text == 'success':
                # Extract session ID
                session_elem = root.find('PHPSESSID')
                if session_elem is not None:
                    self.session_id = session_elem.text
                    self.authenticated = True
                    return True
            
            raise SMTAuthenticationError("Login failed")
            
        except SMTAPIError as e:
            raise SMTAuthenticationError(f"Authentication error: {e}")
    
    def logout(self) -> bool:
        """
        End the current session.
        
        Returns:
            True if logout successful
        """
        if not self.authenticated:
            return True
        
        try:
            root = self._make_request(action='logout')
            logout_result = root.find('logout')
            
            if logout_result is not None and logout_result.text == 'success':
                self.session_id = None
                self.authenticated = False
                return True
            
            return False
            
        except SMTAPIError:
            return False
    
    def list_nodes(self, job_id: int) -> List[NodeMetadata]:
        """
        List all nodes for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            List of NodeMetadata objects
            
        Raises:
            SMTAPIError: If not authenticated or request fails
        """
        if not self.authenticated:
            raise SMTAPIError("Not authenticated. Call login() first.")
        
        root = self._make_request(
            action='listNode',
            params={'jobID': str(job_id)}
        )
        
        nodes = []
        nodes_elem = root.find('nodes')
        
        if nodes_elem is not None:
            for node_elem in nodes_elem.findall('node'):
                node_id = int(node_elem.find('nodeID').text)
                phy_id = int(node_elem.find('phyID').text)
                name = node_elem.find('name').text or ""
                
                # Extract location from name or use name as location
                location = name
                
                created_str = node_elem.find('created').text
                modified_str = node_elem.find('modified').text
                
                created = datetime.strptime(created_str, '%Y-%m-%d %H:%M:%S') if created_str else None
                modified = datetime.strptime(modified_str, '%Y-%m-%d %H:%M:%S') if modified_str else None
                
                nodes.append(NodeMetadata(
                    node_id=node_id,
                    physical_id=phy_id,
                    name=name,
                    location=location,
                    created=created,
                    modified=modified
                ))
        
        return nodes
    
    def list_sensors(self, node_id: int) -> List[SensorMetadata]:
        """
        List all sensors attached to a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            List of SensorMetadata objects
            
        Raises:
            SMTAPIError: If not authenticated or request fails
        """
        if not self.authenticated:
            raise SMTAPIError("Not authenticated. Call login() first.")
        
        root = self._make_request(
            action='listSensor',
            params={'nodeID': str(node_id)}
        )
        
        sensors = []
        sensors_elem = root.find('sensors')
        
        if sensors_elem is not None:
            for sensor_elem in sensors_elem.findall('sensor'):
                sensor_id = int(sensor_elem.find('sensorID').text)
                name = sensor_elem.find('name').text or ""
                sensor_type_name = sensor_elem.find('sensorTypeName').text or ""
                input_channel = int(sensor_elem.find('input').text) if sensor_elem.find('input').text else None
                
                created_str = sensor_elem.find('created').text
                modified_str = sensor_elem.find('modified').text
                
                created = datetime.strptime(created_str, '%Y-%m-%d %H:%M:%S') if created_str else None
                modified = datetime.strptime(modified_str, '%Y-%m-%d %H:%M:%S') if modified_str else None
                
                # Determine unit based on sensor type
                unit = self._determine_unit(sensor_type_name)
                
                # Get location from node name (need to cache this)
                location = name  # Fallback to sensor name
                
                sensors.append(SensorMetadata(
                    sensor_id=sensor_id,
                    name=name,
                    sensor_type=sensor_type_name.lower(),
                    location=location,
                    unit=unit,
                    node_id=node_id,
                    input_channel=input_channel,
                    created=created,
                    modified=modified
                ))
        
        return sensors
    
    def get_sensor_data(
        self,
        sensor_id: int,
        start_date: datetime,
        end_date: datetime
    ) -> List[SensorReading]:
        """
        Retrieve sensor data for a date range.
        
        Args:
            sensor_id: Sensor identifier
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            List of SensorReading objects
            
        Raises:
            SMTAPIError: If not authenticated or request fails
        """
        if not self.authenticated:
            raise SMTAPIError("Not authenticated. Call login() first.")
        
        # Format dates for API (YYYY-MM-DD)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        root = self._make_request(
            action='listSensorData',
            params={
                'sensorID': str(sensor_id),
                'startDate': start_str,
                'endDate': end_str
            }
        )
        
        readings = []
        readings_elem = root.find('readings')
        
        if readings_elem is not None:
            for reading_elem in readings_elem.findall('reading'):
                raw_value = float(reading_elem.find('raw').text) if reading_elem.find('raw').text else None
                eng_value = float(reading_elem.find('engUnit').text) if reading_elem.find('engUnit').text else None
                timestamp_str = reading_elem.find('timestamp').text
                
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                
                # Use engineering unit value if available, otherwise raw
                value = eng_value if eng_value is not None else raw_value
                
                # Determine quality flag (0 for valid data)
                quality_flag = 0 if value is not None else 2
                
                # Unit would need to be determined from sensor metadata
                unit = "unknown"
                
                readings.append(SensorReading(
                    timestamp=timestamp,
                    value=value or 0.0,
                    unit=unit,
                    raw_value=raw_value,
                    quality_flag=quality_flag
                ))
        
        return readings
    
    @staticmethod
    def _determine_unit(sensor_type: str) -> str:
        """
        Determine unit based on sensor type name.
        
        Args:
            sensor_type: Sensor type name
            
        Returns:
            Unit string
        """
        sensor_type_lower = sensor_type.lower()
        
        if 'temp' in sensor_type_lower or 'temperature' in sensor_type_lower:
            return '°C'
        elif 'humidity' in sensor_type_lower or 'rh' in sensor_type_lower:
            return '%'
        elif 'co2' in sensor_type_lower or 'carbon' in sensor_type_lower:
            return 'ppm'
        elif 'moisture' in sensor_type_lower or 'mc' in sensor_type_lower:
            return '%'
        elif 'load' in sensor_type_lower or 'force' in sensor_type_lower:
            return 'N'
        elif 'strain' in sensor_type_lower:
            return 'με'
        else:
            return 'units'
    
    def __enter__(self):
        """Context manager entry."""
        self.login()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.logout()
        return False