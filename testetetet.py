import pygame
import pygame_gui

pygame.init()

# Set up the display
display_surface = pygame.display.set_mode((400, 400))
pygame.display.set_caption('Switchable Button')

# Create the pygame_gui.UIManager object
manager = pygame_gui.UIManager((400, 400))

# Create a Boolean variable to track the button's state
button_state = False

# Create the button
button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((50, 50), (100, 50)),
    text='Off',
    manager=manager
)

# Define the function to toggle the button state and update the text on the button
def toggle_button():
    global button_state
    button_state = not button_state
    if button_state:
        button.set_text('On')
    else:
        button.set_text('Off')

# The main game loop
clock = pygame.time.Clock()
running = True
while running:
    time_delta = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Check for the UI_BUTTON_PRESSED event and call toggle_button() function
        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == button:
                    toggle_button()

        # Pass the event to the pygame_gui.UIManager object
        manager.process_events(event)

    # Update the pygame_gui.UIManager object
    manager.update(time_delta)

    # Draw the pygame_gui.UIManager object
    manager.draw_ui(display_surface)

    pygame.display.update()

pygame.quit()
