"""
Unit tests for logging_utils module.
"""

import os
import sys
import tempfile
import shutil
import unittest
import logging
from pathlib import Path
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nlp_templates.utils.logging_utils import Logger, get_logger


class TestLoggerBasics(unittest.TestCase):
    """Test basic Logger functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear cached loggers before each test
        Logger.clear_loggers()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after test."""
        Logger.clear_loggers()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_logger_creation(self):
        """Test basic logger creation."""
        logger = get_logger("test_logger")
        self.assertIsNotNone(logger)
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "test_logger")

    def test_logger_level(self):
        """Test logger log level setting."""
        logger = Logger.get_logger("level_test", log_level="DEBUG")
        self.assertEqual(logger.level, logging.DEBUG)

        logger2 = Logger.get_logger("level_test2", log_level="WARNING")
        self.assertEqual(logger2.level, logging.WARNING)

    def test_logger_singleton(self):
        """Test that same logger name returns same instance."""
        logger1 = get_logger("singleton_test")
        logger2 = get_logger("singleton_test")
        self.assertIs(logger1, logger2)

    def test_different_loggers(self):
        """Test that different logger names create different instances."""
        logger1 = get_logger("logger_1")
        logger2 = get_logger("logger_2")
        self.assertIsNot(logger1, logger2)

    def test_console_logger_creation(self):
        """Test logger with only console output."""
        logger = Logger.get_logger(
            "console_test", console_output=True, log_file=None
        )
        # Should have exactly one handler (console)
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)

    def test_file_logger_creation(self):
        """Test logger with file output."""
        log_file = os.path.join(self.temp_dir, "test.log")
        logger = Logger.get_logger(
            "file_test", console_output=False, log_file=log_file
        )
        # Should have exactly one handler (file)
        self.assertEqual(len(logger.handlers), 1)
        self.assertIsInstance(logger.handlers[0], logging.FileHandler)

    def test_both_outputs_logger(self):
        """Test logger with both console and file output."""
        log_file = os.path.join(self.temp_dir, "both.log")
        logger = Logger.get_logger(
            "both_test", console_output=True, log_file=log_file
        )
        # Should have two handlers (console + file)
        self.assertEqual(len(logger.handlers), 2)

    def test_log_file_creation(self):
        """Test that log file is created."""
        log_file = os.path.join(self.temp_dir, "created.log")
        logger = Logger.get_logger("file_creation_test", log_file=log_file)
        logger.info("Test message")

        self.assertTrue(os.path.exists(log_file))

    def test_nested_directory_creation(self):
        """Test that nested directories are created for log file."""
        log_file = os.path.join(self.temp_dir, "nested", "dirs", "test.log")
        logger = Logger.get_logger(
            "nested_test", log_file=log_file, console_output=False
        )
        logger.info("Test message")

        self.assertTrue(os.path.exists(log_file))
        self.assertTrue(os.path.isdir(os.path.dirname(log_file)))

    def test_log_message_format(self):
        """Test that log messages have correct format."""
        log_file = os.path.join(self.temp_dir, "format.log")
        logger = Logger.get_logger(
            "format_test",
            log_file=log_file,
            console_output=False,
            log_level="INFO",
        )
        logger.info("Test message")

        with open(log_file, "r") as f:
            content = f.read()

        # Check that log file contains expected elements
        self.assertIn("format_test", content)
        self.assertIn("INFO", content)
        self.assertIn("Test message", content)


class TestLoggerLevels(unittest.TestCase):
    """Test different log levels."""

    def setUp(self):
        """Set up test fixtures."""
        Logger.clear_loggers()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after test."""
        Logger.clear_loggers()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_debug_level(self):
        """Test DEBUG level logging."""
        log_file = os.path.join(self.temp_dir, "debug.log")
        logger = Logger.get_logger(
            "debug_test",
            log_level="DEBUG",
            log_file=log_file,
            console_output=False,
        )
        logger.debug("Debug message")

        with open(log_file, "r") as f:
            content = f.read()
        self.assertIn("Debug message", content)

    def test_info_level(self):
        """Test INFO level logging."""
        log_file = os.path.join(self.temp_dir, "info.log")
        logger = Logger.get_logger(
            "info_test",
            log_level="INFO",
            log_file=log_file,
            console_output=False,
        )
        logger.info("Info message")

        with open(log_file, "r") as f:
            content = f.read()
        self.assertIn("Info message", content)

    def test_warning_level(self):
        """Test WARNING level logging."""
        log_file = os.path.join(self.temp_dir, "warning.log")
        logger = Logger.get_logger(
            "warning_test",
            log_level="WARNING",
            log_file=log_file,
            console_output=False,
        )
        logger.warning("Warning message")

        with open(log_file, "r") as f:
            content = f.read()
        self.assertIn("Warning message", content)

    def test_error_level(self):
        """Test ERROR level logging."""
        log_file = os.path.join(self.temp_dir, "error.log")
        logger = Logger.get_logger(
            "error_test",
            log_level="ERROR",
            log_file=log_file,
            console_output=False,
        )
        logger.error("Error message")

        with open(log_file, "r") as f:
            content = f.read()
        self.assertIn("Error message", content)

    def test_critical_level(self):
        """Test CRITICAL level logging."""
        log_file = os.path.join(self.temp_dir, "critical.log")
        logger = Logger.get_logger(
            "critical_test",
            log_level="CRITICAL",
            log_file=log_file,
            console_output=False,
        )
        logger.critical("Critical message")

        with open(log_file, "r") as f:
            content = f.read()
        self.assertIn("Critical message", content)

    def test_level_filtering(self):
        """Test that lower level messages are filtered."""
        log_file = os.path.join(self.temp_dir, "filtering.log")
        logger = Logger.get_logger(
            "filter_test",
            log_level="WARNING",
            log_file=log_file,
            console_output=False,
        )
        logger.debug("Debug (should not appear)")
        logger.info("Info (should not appear)")
        logger.warning("Warning (should appear)")
        logger.error("Error (should appear)")

        with open(log_file, "r") as f:
            content = f.read()

        self.assertNotIn("Debug (should not appear)", content)
        self.assertNotIn("Info (should not appear)", content)
        self.assertIn("Warning (should appear)", content)
        self.assertIn("Error (should appear)", content)


class TestGetLoggerFunction(unittest.TestCase):
    """Test get_logger convenience function."""

    def setUp(self):
        """Set up test fixtures."""
        Logger.clear_loggers()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after test."""
        Logger.clear_loggers()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_get_logger_defaults(self):
        """Test get_logger with default parameters."""
        logger = get_logger()
        self.assertEqual(logger.name, "nlp_templates")
        self.assertEqual(logger.level, logging.INFO)

    def test_get_logger_custom_name(self):
        """Test get_logger with custom name."""
        logger = get_logger("custom_module")
        self.assertEqual(logger.name, "custom_module")

    def test_get_logger_custom_level(self):
        """Test get_logger with custom log level."""
        logger = get_logger("level_module", log_level="DEBUG")
        self.assertEqual(logger.level, logging.DEBUG)

    def test_get_logger_with_file(self):
        """Test get_logger with log file."""
        log_file = os.path.join(self.temp_dir, "convenience.log")
        logger = get_logger("file_module", log_file=log_file)
        logger.info("Test message")

        self.assertTrue(os.path.exists(log_file))


class TestRootLogger(unittest.TestCase):
    """Test root logger setup."""

    def setUp(self):
        """Set up test fixtures."""
        Logger.clear_loggers()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after test."""
        Logger.clear_loggers()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_setup_root_logger(self):
        """Test root logger setup."""
        root_logger = Logger.setup_root_logger(log_level="DEBUG")
        self.assertIsNotNone(root_logger)
        self.assertEqual(root_logger.name, "nlp_templates")
        self.assertEqual(root_logger.level, logging.DEBUG)

    def test_root_logger_with_file(self):
        """Test root logger with file output."""
        log_file = os.path.join(self.temp_dir, "root.log")
        root_logger = Logger.setup_root_logger(
            log_level="INFO", log_file=log_file
        )
        root_logger.info("Root logger message")

        self.assertTrue(os.path.exists(log_file))

    def test_root_logger_file_content(self):
        """Test that root logger writes to file correctly."""
        log_file = os.path.join(self.temp_dir, "root_content.log")
        root_logger = Logger.setup_root_logger(
            log_level="INFO", log_file=log_file
        )
        root_logger.info("Test root message")

        with open(log_file, "r") as f:
            content = f.read()

        self.assertIn("Test root message", content)


class TestLoggerUtility(unittest.TestCase):
    """Test logger utility methods."""

    def setUp(self):
        """Set up test fixtures."""
        Logger.clear_loggers()

    def tearDown(self):
        """Clean up after test."""
        Logger.clear_loggers()

    def test_disable_logger(self):
        """Test disabling a logger."""
        logger = get_logger("disable_test")
        Logger.disable_logger("disable_test")

        # Logger should still exist but at CRITICAL level
        self.assertEqual(logger.level, logging.CRITICAL)

    def test_clear_loggers(self):
        """Test clearing all loggers."""
        logger1 = get_logger("clear_test_1")
        logger2 = get_logger("clear_test_2")

        # Verify loggers exist and have handlers
        self.assertGreater(len(logger1.handlers), 0)
        self.assertGreater(len(logger2.handlers), 0)

        # Clear loggers
        Logger.clear_loggers()

        # Verify cache is cleared
        self.assertEqual(len(Logger._loggers), 0)


class TestLoggerIntegration(unittest.TestCase):
    """Integration tests for logger."""

    def setUp(self):
        """Set up test fixtures."""
        Logger.clear_loggers()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after test."""
        Logger.clear_loggers()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_multiple_loggers_different_levels(self):
        """Test multiple loggers with different levels."""
        log_file_1 = os.path.join(self.temp_dir, "multi_1.log")
        log_file_2 = os.path.join(self.temp_dir, "multi_2.log")

        logger1 = Logger.get_logger(
            "multi_1",
            log_level="DEBUG",
            log_file=log_file_1,
            console_output=False,
        )
        logger2 = Logger.get_logger(
            "multi_2",
            log_level="WARNING",
            log_file=log_file_2,
            console_output=False,
        )

        logger1.debug("Debug message 1")
        logger1.info("Info message 1")

        logger2.debug("Debug message 2 (should not appear)")
        logger2.warning("Warning message 2")

        # Check logger1 log file
        with open(log_file_1, "r") as f:
            content1 = f.read()
        self.assertIn("Debug message 1", content1)
        self.assertIn("Info message 1", content1)

        # Check logger2 log file
        with open(log_file_2, "r") as f:
            content2 = f.read()
        self.assertNotIn("Debug message 2", content2)
        self.assertIn("Warning message 2", content2)

    def test_custom_format(self):
        """Test logger with custom format."""
        log_file = os.path.join(self.temp_dir, "custom_format.log")
        custom_format = "%(levelname)s | %(name)s | %(message)s"

        logger = Logger.get_logger(
            "custom_format_test",
            log_file=log_file,
            console_output=False,
            log_format=custom_format,
        )
        logger.info("Custom format message")

        with open(log_file, "r") as f:
            content = f.read()

        # Check that custom format is used
        self.assertIn("INFO |", content)
        self.assertIn("custom_format_test |", content)

    def test_logger_persistence(self):
        """Test that logger persists messages across calls."""
        log_file = os.path.join(self.temp_dir, "persistence.log")
        logger = Logger.get_logger(
            "persist_test", log_file=log_file, console_output=False
        )

        logger.info("Message 1")
        logger.info("Message 2")
        logger.info("Message 3")

        with open(log_file, "r") as f:
            content = f.read()

        self.assertIn("Message 1", content)
        self.assertIn("Message 2", content)
        self.assertIn("Message 3", content)


def run_tests():
    """Run all tests and print results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLoggerBasics))
    suite.addTests(loader.loadTestsFromTestCase(TestLoggerLevels))
    suite.addTests(loader.loadTestsFromTestCase(TestGetLoggerFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestRootLogger))
    suite.addTests(loader.loadTestsFromTestCase(TestLoggerUtility))
    suite.addTests(loader.loadTestsFromTestCase(TestLoggerIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
